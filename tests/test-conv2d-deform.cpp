#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct conv2d_deform_case {
    int iw = 5;
    int ih = 4;
    int ic = 2;
    int n  = 2;
    int kw = 3;
    int kh = 3;
    int oc = 3;
    int stride_x = 1;
    int stride_y = 1;
    int pad_x = 1;
    int pad_y = 1;
};

static int conv_out_size(int input, int kernel, int stride, int pad) {
    return (input + 2 * pad - kernel) / stride + 1;
}

static size_t idx4(int x, int y, int c, int n, int ne0, int ne1, int ne2) {
    return (size_t) x + (size_t) ne0 * ((size_t) y + (size_t) ne1 * ((size_t) c + (size_t) ne2 * (size_t) n));
}

static std::vector<float> make_data(size_t n, float scale, float bias) {
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = bias + scale * (0.7f * std::sin(0.17f * (float) (i + 1)) +
                                  0.3f * std::cos(0.11f * (float) (i + 3)));
    }
    return data;
}

static std::vector<float> make_offsets(size_t n) {
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = 0.45f * std::sin(0.31f * (float) (i + 5));
    }
    return data;
}

static std::vector<float> make_mask(size_t n) {
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = 0.55f + 0.35f * std::cos(0.23f * (float) (i + 7));
    }
    return data;
}

static float bilinear_sample(
        const std::vector<float> & input,
        const conv2d_deform_case & tc,
        int batch,
        int channel,
        float x,
        float y) {
    const int x0 = (int) std::floor(x);
    const int y0 = (int) std::floor(y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float dx = x - (float) x0;
    const float dy = y - (float) y0;

    auto at = [&](int xx, int yy) {
        if (xx < 0 || xx >= tc.iw || yy < 0 || yy >= tc.ih) {
            return 0.0f;
        }
        return input[idx4(xx, yy, channel, batch, tc.iw, tc.ih, tc.ic)];
    };

    const float v00 = at(x0, y0);
    const float v01 = at(x1, y0);
    const float v10 = at(x0, y1);
    const float v11 = at(x1, y1);

    const float v0 = v00 * (1.0f - dx) + v01 * dx;
    const float v1 = v10 * (1.0f - dx) + v11 * dx;
    return v0 * (1.0f - dy) + v1 * dy;
}

static std::vector<float> conv2d_deform_reference(
        const conv2d_deform_case & tc,
        const std::vector<float> & kernel,
        const std::vector<float> & input,
        const std::vector<float> & offset,
        const std::vector<float> & mask) {
    const int ow = conv_out_size(tc.iw, tc.kw, tc.stride_x, tc.pad_x);
    const int oh = conv_out_size(tc.ih, tc.kh, tc.stride_y, tc.pad_y);
    std::vector<float> output((size_t) ow * oh * tc.oc * tc.n, 0.0f);

    for (int n = 0; n < tc.n; ++n) {
        for (int oc = 0; oc < tc.oc; ++oc) {
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < tc.kh; ++ky) {
                        for (int kx = 0; kx < tc.kw; ++kx) {
                            const int k = ky * tc.kw + kx;
                            const float off_y = offset[idx4(ox, oy, 2 * k + 0, n, ow, oh, 2 * tc.kw * tc.kh)];
                            const float off_x = offset[idx4(ox, oy, 2 * k + 1, n, ow, oh, 2 * tc.kw * tc.kh)];
                            const float sx = (float) (ox * tc.stride_x + kx - tc.pad_x) + off_x;
                            const float sy = (float) (oy * tc.stride_y + ky - tc.pad_y) + off_y;
                            if (sx <= -1.0f || sx >= (float) tc.iw || sy <= -1.0f || sy >= (float) tc.ih) {
                                continue;
                            }

                            const float m = mask[idx4(ox, oy, k, n, ow, oh, tc.kw * tc.kh)];
                            for (int ic = 0; ic < tc.ic; ++ic) {
                                const float v = bilinear_sample(input, tc, n, ic, sx, sy);
                                const float w = kernel[idx4(kx, ky, ic, oc, tc.kw, tc.kh, tc.ic)];
                                acc += v * m * w;
                            }
                        }
                    }
                    output[idx4(ox, oy, oc, n, ow, oh, tc.oc)] = acc;
                }
            }
        }
    }

    return output;
}

static std::vector<uint8_t> make_kernel_payload(
        ggml_type type,
        const std::vector<float> & kernel_f32,
        std::vector<float> & kernel_ref) {
    kernel_ref = kernel_f32;
    if (type == GGML_TYPE_F32) {
        std::vector<uint8_t> payload(kernel_f32.size() * sizeof(float));
        std::memcpy(payload.data(), kernel_f32.data(), payload.size());
        return payload;
    }

    GGML_ASSERT(type == GGML_TYPE_F16);
    std::vector<ggml_fp16_t> kernel_f16(kernel_f32.size());
    ggml_fp32_to_fp16_row(kernel_f32.data(), kernel_f16.data(), kernel_f32.size());
    ggml_fp16_to_fp32_row(kernel_f16.data(), kernel_ref.data(), kernel_ref.size());

    std::vector<uint8_t> payload(kernel_f16.size() * sizeof(ggml_fp16_t));
    std::memcpy(payload.data(), kernel_f16.data(), payload.size());
    return payload;
}

static bool check_close(
        const std::vector<float> & got,
        const std::vector<float> & expected,
        float tol,
        const char * label) {
    if (got.size() != expected.size()) {
        std::printf("%s: size mismatch got=%zu expected=%zu\n", label, got.size(), expected.size());
        return false;
    }

    float max_abs = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        if (diff > max_abs) {
            max_abs = diff;
            max_idx = i;
        }
    }

    if (max_abs > tol) {
        std::printf("%s: max_abs=%g at %zu got=%g expected=%g tol=%g\n",
                label, max_abs, max_idx, got[max_idx], expected[max_idx], tol);
        return false;
    }

    std::printf("%s: max_abs=%g\n", label, max_abs);
    return true;
}

static bool run_backend_case(
        ggml_backend_t backend,
        const char * backend_name,
        ggml_type kernel_type,
        const conv2d_deform_case & tc,
        const std::vector<float> & kernel_f32,
        const std::vector<float> & input,
        const std::vector<float> & offset,
        const std::vector<float> & mask,
        bool & ran) {
    const int ow = conv_out_size(tc.iw, tc.kw, tc.stride_x, tc.pad_x);
    const int oh = conv_out_size(tc.ih, tc.kh, tc.stride_y, tc.pad_y);

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr{ggml_init(params)};
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * kernel = ggml_new_tensor_4d(ctx, kernel_type, tc.kw, tc.kh, tc.ic, tc.oc);
    ggml_tensor * src    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.iw, tc.ih, tc.ic, tc.n);
    ggml_tensor * off    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ow, oh, 2 * tc.kw * tc.kh, tc.n);
    ggml_tensor * msk    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ow, oh, tc.kw * tc.kh, tc.n);
    ggml_tensor * res    = ggml_conv_2d_deform(ctx, kernel, src, off, msk, tc.stride_x, tc.stride_y, tc.pad_x, tc.pad_y);

    if (!ggml_backend_supports_op(backend, res)) {
        std::printf("%s %s: skipped, backend does not support conv_2d_deform\n",
                backend_name, kernel_type == GGML_TYPE_F16 ? "f16" : "f32");
        ran = false;
        return true;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);

    ggml_backend_buffer_ptr buffer{ggml_backend_alloc_ctx_tensors(ctx, backend)};
    if (!buffer) {
        std::printf("%s: failed to allocate backend tensor buffer\n", backend_name);
        ran = false;
        return false;
    }

    std::vector<float> kernel_ref;
    const std::vector<uint8_t> kernel_payload = make_kernel_payload(kernel_type, kernel_f32, kernel_ref);
    ggml_backend_tensor_set(kernel, kernel_payload.data(), 0, kernel_payload.size());
    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(off, offset.data(), 0, offset.size() * sizeof(float));
    ggml_backend_tensor_set(msk, mask.data(), 0, mask.size() * sizeof(float));

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, 2);
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::printf("%s: graph compute failed with status=%s\n", backend_name, ggml_status_to_string(status));
        ran = true;
        return false;
    }

    std::vector<float> got(ggml_nelements(res));
    ggml_backend_tensor_get(res, got.data(), 0, got.size() * sizeof(float));

    const std::vector<float> expected = conv2d_deform_reference(tc, kernel_ref, input, offset, mask);
    const float tol = kernel_type == GGML_TYPE_F16 ? 2e-3f : 1e-3f;

    std::string label = std::string(backend_name) + " conv_2d_deform " +
            (kernel_type == GGML_TYPE_F16 ? "f16" : "f32");
    ran = true;
    return check_close(got, expected, tol, label.c_str());
}

int main() {
    ggml_time_init();
    ggml_backend_load_all();

    const conv2d_deform_case tc;
    const int ow = conv_out_size(tc.iw, tc.kw, tc.stride_x, tc.pad_x);
    const int oh = conv_out_size(tc.ih, tc.kh, tc.stride_y, tc.pad_y);

    const std::vector<float> kernel = make_data((size_t) tc.kw * tc.kh * tc.ic * tc.oc, 0.35f, -0.05f);
    const std::vector<float> input  = make_data((size_t) tc.iw * tc.ih * tc.ic * tc.n, 0.85f, 0.10f);
    const std::vector<float> offset = make_offsets((size_t) ow * oh * 2 * tc.kw * tc.kh * tc.n);
    const std::vector<float> mask   = make_mask((size_t) ow * oh * tc.kw * tc.kh * tc.n);

    bool ok = true;
    int non_cpu_ran = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_ptr backend_ptr{ggml_backend_dev_init(dev, nullptr)};
        ggml_backend_t backend = backend_ptr.get();
        if (backend == nullptr) {
            std::printf("%s: skipped, failed to initialize backend\n", ggml_backend_dev_name(dev));
            continue;
        }

        const char * backend_name = ggml_backend_name(backend);
        for (ggml_type kernel_type : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
            bool ran = false;
            ok = run_backend_case(backend, backend_name, kernel_type, tc, kernel, input, offset, mask, ran) && ok;
            if (ran && !ggml_backend_is_cpu(backend)) {
                ++non_cpu_ran;
            }
        }
    }

    if (non_cpu_ran == 0) {
        std::printf("no non-CPU backend ran conv_2d_deform in this build\n");
    }

    return ok ? 0 : 1;
}
