#include "conv3d.cuh"
#include "convert.cuh"

struct conv3d_params {
    int64_t IW, IH, ID;
    int64_t OW, OH, OD;
    int64_t KW, KH, KD;
    int64_t ST_X, ST_Y, ST_Z;
    int64_t PD_X, PD_Y, PD_Z;
    int64_t DL_X, DL_Y, DL_Z;
    int64_t IC, OC, B;
    int64_t nb00, nb01, nb02, nb03;
    int64_t nb10, nb11, nb12, nb13;
    int64_t nb0,  nb1,  nb2,  nb3;
    int64_t TOTAL;
};

template <typename T>
static __global__ void conv3d_kernel(
        const char * __restrict__ kernel,
        const char * __restrict__ input,
        char       * __restrict__ output,
        const conv3d_params P) {
    const int64_t spatial_idx = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (spatial_idx >= P.OW * P.OH * P.OD) {
        return;
    }

    const int64_t ow = spatial_idx % P.OW;
    const int64_t oh = (spatial_idx / P.OW) % P.OH;
    const int64_t od = spatial_idx / (P.OW * P.OH);
    const int64_t oc = blockIdx.y;
    const int64_t b  = blockIdx.z;

    const int64_t iw_base = ow * P.ST_X - P.PD_X;
    const int64_t ih_base = oh * P.ST_Y - P.PD_Y;
    const int64_t id_base = od * P.ST_Z - P.PD_Z;

    float acc = 0.0f;
    for (int64_t ic = 0; ic < P.IC; ++ic) {
        const int64_t input_cn = b * P.IC + ic;
        const int64_t kernel_cn = oc * P.IC + ic;

        for (int64_t kz = 0; kz < P.KD; ++kz) {
            const int64_t id = id_base + kz * P.DL_Z;
            if (id < 0 || id >= P.ID) {
                continue;
            }

            for (int64_t ky = 0; ky < P.KH; ++ky) {
                const int64_t ih = ih_base + ky * P.DL_Y;
                if (ih < 0 || ih >= P.IH) {
                    continue;
                }

                for (int64_t kx = 0; kx < P.KW; ++kx) {
                    const int64_t iw = iw_base + kx * P.DL_X;
                    if (iw < 0 || iw >= P.IW) {
                        continue;
                    }

                    const int64_t kernel_offset = kx * P.nb00 + ky * P.nb01 + kz * P.nb02 + kernel_cn * P.nb03;
                    const int64_t input_offset  = iw * P.nb10 + ih * P.nb11 + id * P.nb12 + input_cn  * P.nb13;
                    const float kv = ggml_cuda_cast<float>(*(const T *) (kernel + kernel_offset));
                    const float xv = *(const float *) (input + input_offset);
                    acc += kv * xv;
                }
            }
        }
    }

    const int64_t output_cn = b * P.OC + oc;
    const int64_t output_offset = ow * P.nb0 + oh * P.nb1 + od * P.nb2 + output_cn * P.nb3;
    *(float *) (output + output_offset) = acc;
}

template <typename T>
static void conv3d_cuda(
        const char * kernel,
        const char * input,
        char * output,
        const conv3d_params P,
        cudaStream_t stream) {
    const int blocks = (int) ((P.OW * P.OH * P.OD + CUDA_CONV3D_BLOCK_SIZE - 1) / CUDA_CONV3D_BLOCK_SIZE);
    const dim3 grid(blocks, (uint32_t) P.OC, (uint32_t) P.B);
    conv3d_kernel<T><<<grid, CUDA_CONV3D_BLOCK_SIZE, 0, stream>>>(kernel, input, output, P);
}

void ggml_cuda_op_conv3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t * p = (const int32_t *) dst->op_params;
    const int32_t IC = p[9];
    const int32_t B  = p[10];
    const int32_t OC = p[11];

    GGML_ASSERT(kernel->ne[3] == (int64_t) IC * OC);
    GGML_ASSERT(input->ne[3]  == (int64_t) IC * B);
    GGML_ASSERT(dst->ne[3]    == (int64_t) OC * B);

    const conv3d_params params = {
        /*.IW =*/ input->ne[0],
        /*.IH =*/ input->ne[1],
        /*.ID =*/ input->ne[2],
        /*.OW =*/ dst->ne[0],
        /*.OH =*/ dst->ne[1],
        /*.OD =*/ dst->ne[2],
        /*.KW =*/ kernel->ne[0],
        /*.KH =*/ kernel->ne[1],
        /*.KD =*/ kernel->ne[2],
        /*.ST_X =*/ p[0],
        /*.ST_Y =*/ p[1],
        /*.ST_Z =*/ p[2],
        /*.PD_X =*/ p[3],
        /*.PD_Y =*/ p[4],
        /*.PD_Z =*/ p[5],
        /*.DL_X =*/ p[6],
        /*.DL_Y =*/ p[7],
        /*.DL_Z =*/ p[8],
        /*.IC =*/ IC,
        /*.OC =*/ OC,
        /*.B =*/ B,
        /*.nb00 =*/ (int64_t) kernel->nb[0],
        /*.nb01 =*/ (int64_t) kernel->nb[1],
        /*.nb02 =*/ (int64_t) kernel->nb[2],
        /*.nb03 =*/ (int64_t) kernel->nb[3],
        /*.nb10 =*/ (int64_t) input->nb[0],
        /*.nb11 =*/ (int64_t) input->nb[1],
        /*.nb12 =*/ (int64_t) input->nb[2],
        /*.nb13 =*/ (int64_t) input->nb[3],
        /*.nb0 =*/ (int64_t) dst->nb[0],
        /*.nb1 =*/ (int64_t) dst->nb[1],
        /*.nb2 =*/ (int64_t) dst->nb[2],
        /*.nb3 =*/ (int64_t) dst->nb[3],
        /*.TOTAL =*/ ggml_nelements(dst),
    };

    cudaStream_t stream = ctx.stream();
    if (kernel->type == GGML_TYPE_F16) {
        conv3d_cuda<half>((const char *) kernel->data, (const char *) input->data, (char *) dst->data, params, stream);
    } else {
        conv3d_cuda<float>((const char *) kernel->data, (const char *) input->data, (char *) dst->data, params, stream);
    }
}
