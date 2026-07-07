#include "conv2d-deform.cuh"
#include "convert.cuh"

struct deform_conv_params {
    int64_t iw, ih;
    int64_t ow, oh;
    int64_t kw, kh;
    int64_t stride_x, stride_y;
    int64_t pad_x, pad_y;
    int64_t ic, oc;
    int64_t batch;
    int64_t total;
};

struct deform_whcn_layout {
    __device__ static int64_t input_index(int64_t n, int64_t c, int64_t y, int64_t x, const deform_conv_params & p) {
        return n * (p.ic * p.iw * p.ih) + c * (p.iw * p.ih) + y * p.iw + x;
    }

    __device__ static int64_t kernel_index(int64_t oc, int64_t ic, int64_t ky, int64_t kx, const deform_conv_params & p) {
        return oc * (p.ic * p.kh * p.kw) + ic * (p.kh * p.kw) + ky * p.kw + kx;
    }

    __device__ static int64_t offset_index(int64_t n, int64_t c, int64_t y, int64_t x, const deform_conv_params & p) {
        return n * (2 * p.kw * p.kh * p.ow * p.oh) + c * (p.ow * p.oh) + y * p.ow + x;
    }

    __device__ static int64_t mask_index(int64_t n, int64_t c, int64_t y, int64_t x, const deform_conv_params & p) {
        return n * (p.kw * p.kh * p.ow * p.oh) + c * (p.ow * p.oh) + y * p.ow + x;
    }

    __device__ static int64_t output_index(int64_t n, int64_t c, int64_t y, int64_t x, const deform_conv_params & p) {
        return n * (p.oc * p.ow * p.oh) + c * (p.ow * p.oh) + y * p.ow + x;
    }

    __device__ static void unpack(int64_t idx, const deform_conv_params & p, int64_t & n, int64_t & oc, int64_t & y, int64_t & x) {
        x  = idx % p.ow;
        y  = (idx / p.ow) % p.oh;
        oc = (idx / (p.ow * p.oh)) % p.oc;
        n  = idx / (p.ow * p.oh * p.oc);
    }
};

struct deform_cwhn_layout {
    __device__ static int64_t input_index(int64_t n, int64_t c, int64_t y, int64_t x, const deform_conv_params & p) {
        return n * (p.ic * p.iw * p.ih) + y * (p.iw * p.ic) + x * p.ic + c;
    }

    __device__ static int64_t kernel_index(int64_t oc, int64_t ic, int64_t ky, int64_t kx, const deform_conv_params & p) {
        return oc * (p.ic * p.kw * p.kh) + ky * (p.kw * p.ic) + kx * p.ic + ic;
    }

    __device__ static int64_t offset_index(int64_t n, int64_t c, int64_t y, int64_t x, const deform_conv_params & p) {
        const int64_t cc = 2 * p.kw * p.kh;
        return n * (cc * p.ow * p.oh) + y * (p.ow * cc) + x * cc + c;
    }

    __device__ static int64_t mask_index(int64_t n, int64_t c, int64_t y, int64_t x, const deform_conv_params & p) {
        const int64_t cc = p.kw * p.kh;
        return n * (cc * p.ow * p.oh) + y * (p.ow * cc) + x * cc + c;
    }

    __device__ static int64_t output_index(int64_t n, int64_t c, int64_t y, int64_t x, const deform_conv_params & p) {
        return n * (p.oc * p.ow * p.oh) + y * (p.ow * p.oc) + x * p.oc + c;
    }

    __device__ static void unpack(int64_t idx, const deform_conv_params & p, int64_t & n, int64_t & oc, int64_t & y, int64_t & x) {
        oc = idx % p.oc;
        x  = (idx / p.oc) % p.ow;
        y  = (idx / (p.oc * p.ow)) % p.oh;
        n  = idx / (p.oc * p.ow * p.oh);
    }
};

template <typename Layout>
__device__ float deform_bilinear_sample(
        const float * __restrict__ input,
        const deform_conv_params & p,
        int64_t n,
        int64_t c,
        float x,
        float y) {
    const int64_t x0 = (int64_t) floorf(x);
    const int64_t y0 = (int64_t) floorf(y);
    const int64_t x1 = x0 + 1;
    const int64_t y1 = y0 + 1;
    const float dx = x - (float) x0;
    const float dy = y - (float) y0;

    float v00 = 0.0f;
    float v01 = 0.0f;
    float v10 = 0.0f;
    float v11 = 0.0f;
    if (x0 >= 0 && x0 < p.iw && y0 >= 0 && y0 < p.ih) {
        v00 = input[Layout::input_index(n, c, y0, x0, p)];
    }
    if (x1 >= 0 && x1 < p.iw && y0 >= 0 && y0 < p.ih) {
        v01 = input[Layout::input_index(n, c, y0, x1, p)];
    }
    if (x0 >= 0 && x0 < p.iw && y1 >= 0 && y1 < p.ih) {
        v10 = input[Layout::input_index(n, c, y1, x0, p)];
    }
    if (x1 >= 0 && x1 < p.iw && y1 >= 0 && y1 < p.ih) {
        v11 = input[Layout::input_index(n, c, y1, x1, p)];
    }

    const float v0 = v00 * (1.0f - dx) + v01 * dx;
    const float v1 = v10 * (1.0f - dx) + v11 * dx;
    return v0 * (1.0f - dy) + v1 * dy;
}

template <typename kernel_t, typename Layout>
__global__ void conv2d_deform_kernel(
        const kernel_t * __restrict__ kernel,
        const float * __restrict__ input,
        const float * __restrict__ offset,
        const float * __restrict__ mask,
        float * __restrict__ output,
        const deform_conv_params p) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.total) {
        return;
    }

    int64_t n, oc, out_y, out_x;
    Layout::unpack(idx, p, n, oc, out_y, out_x);

    float acc = 0.0f;
    for (int64_t ky = 0; ky < p.kh; ++ky) {
        for (int64_t kx = 0; kx < p.kw; ++kx) {
            const int64_t k = ky * p.kw + kx;
            const float off_y = offset[Layout::offset_index(n, 2*k + 0, out_y, out_x, p)];
            const float off_x = offset[Layout::offset_index(n, 2*k + 1, out_y, out_x, p)];
            const float sx = (float) (out_x * p.stride_x + kx - p.pad_x) + off_x;
            const float sy = (float) (out_y * p.stride_y + ky - p.pad_y) + off_y;
            if (sx <= -1.0f || sx >= (float) p.iw || sy <= -1.0f || sy >= (float) p.ih) {
                continue;
            }
            const float m = mask == nullptr ? 1.0f : mask[Layout::mask_index(n, k, out_y, out_x, p)];
            for (int64_t ic = 0; ic < p.ic; ++ic) {
                const float v = deform_bilinear_sample<Layout>(input, p, n, ic, sx, sy);
                const float w = ggml_cuda_cast<float>(kernel[Layout::kernel_index(oc, ic, ky, kx, p)]);
                acc += v * m * w;
            }
        }
    }
    output[Layout::output_index(n, oc, out_y, out_x, p)] = acc;
}

template <typename kernel_t, typename Layout>
static void conv2d_deform_cuda(
        const kernel_t * kernel,
        const float * input,
        const float * offset,
        const float * mask,
        float * output,
        const deform_conv_params & p,
        cudaStream_t stream) {
    const int blocks = (p.total + CUDA_CONV2D_DEFORM_BLOCK_SIZE - 1) / CUDA_CONV2D_DEFORM_BLOCK_SIZE;
    conv2d_deform_kernel<kernel_t, Layout><<<blocks, CUDA_CONV2D_DEFORM_BLOCK_SIZE, 0, stream>>>(
        kernel, input, offset, mask, output, p);
}

void ggml_cuda_op_conv2d_deform(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];
    const ggml_tensor * offset = dst->src[2];
    const ggml_tensor * mask   = dst->src[3];

    GGML_ASSERT(input->type == GGML_TYPE_F32 && offset->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mask == nullptr || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16);

    const int32_t * op = (const int32_t *) dst->op_params;
    deform_conv_params p = {
        input->ne[0], input->ne[1],
        dst->ne[0], dst->ne[1],
        kernel->ne[0], kernel->ne[1],
        op[0], op[1],
        op[2], op[3],
        input->ne[2], kernel->ne[3],
        input->ne[3],
        input->ne[3] * kernel->ne[3] * dst->ne[1] * dst->ne[0],
    };

    cudaStream_t stream = ctx.stream();
    const float * input_d = (const float *) input->data;
    const float * offset_d = (const float *) offset->data;
    const float * mask_d = mask == nullptr ? nullptr : (const float *) mask->data;
    float * output_d = (float *) dst->data;

    if (ggml_is_contiguous(input)) {
        if (kernel->type == GGML_TYPE_F16) {
            conv2d_deform_cuda<half, deform_whcn_layout>((const half *) kernel->data, input_d, offset_d, mask_d, output_d, p, stream);
        } else {
            conv2d_deform_cuda<float, deform_whcn_layout>((const float *) kernel->data, input_d, offset_d, mask_d, output_d, p, stream);
        }
    } else if (ggml_is_contiguous_channels(input)) {
        if (kernel->type == GGML_TYPE_F16) {
            conv2d_deform_cuda<half, deform_cwhn_layout>((const half *) kernel->data, input_d, offset_d, mask_d, output_d, p, stream);
        } else {
            conv2d_deform_cuda<float, deform_cwhn_layout>((const float *) kernel->data, input_d, offset_d, mask_d, output_d, p, stream);
        }
    } else {
        GGML_ABORT("Unsupported memory layout for conv_2d_deform");
    }
}
