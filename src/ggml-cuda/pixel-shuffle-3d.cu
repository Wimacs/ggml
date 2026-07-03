#include "pixel-shuffle-3d.cuh"

static __global__ void pixel_shuffle_3d_f32(
        const char * __restrict__ src,
        char       * __restrict__ dst,
        int64_t nb00,
        int64_t nb01,
        int64_t nb02,
        int64_t nb03,
        int64_t nb0,
        int64_t nb1,
        int64_t nb2,
        int64_t nb3,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t scale,
        int64_t total) {
    int64_t linear = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int64_t i0 = linear % ne0;
    linear /= ne0;
    const int64_t i1 = linear % ne1;
    linear /= ne1;
    const int64_t i2 = linear % ne2;
    linear /= ne2;
    const int64_t i3 = linear;

    const int64_t i00 = i0 / scale;
    const int64_t i01 = i1 / scale;
    const int64_t i02 = i2 / scale;
    const int64_t r0 = i0 - i00 * scale;
    const int64_t r1 = i1 - i01 * scale;
    const int64_t r2 = i2 - i02 * scale;
    const int64_t i03 = i3 * scale * scale * scale + r2 * scale * scale + r1 * scale + r0;

    const float * x = (const float *) (src + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
    float * y = (float *) (dst + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
    *y = *x;
}

void ggml_cuda_op_pixel_shuffle_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t scale = ggml_get_op_params_i32(dst, 0);
    const int64_t scale3 = scale * scale * scale;
    GGML_ASSERT(src0->ne[3] == dst->ne[3] * scale3);

    const int64_t total = ggml_nelements(dst);
    const int blocks = (int) ((total + CUDA_PIXEL_SHUFFLE_3D_BLOCK_SIZE - 1) / CUDA_PIXEL_SHUFFLE_3D_BLOCK_SIZE);
    pixel_shuffle_3d_f32<<<blocks, CUDA_PIXEL_SHUFFLE_3D_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const char *) src0->data,
        (char *) dst->data,
        (int64_t) src0->nb[0],
        (int64_t) src0->nb[1],
        (int64_t) src0->nb[2],
        (int64_t) src0->nb[3],
        (int64_t) dst->nb[0],
        (int64_t) dst->nb[1],
        (int64_t) dst->nb[2],
        (int64_t) dst->nb[3],
        dst->ne[0],
        dst->ne[1],
        dst->ne[2],
        scale,
        total);
}
