//
// Created by saul on 2025/8/7.
//
#include <assert.h>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include "my_tools.h"
#include "dma_alloc.h"
#include "im2d_buffer.h"


int main(int n_args, char **args) {
    assert(n_args == 3);
    char *model_path = args[1];
    char *img_path = args[2];

    const char *rga_info = querystring(RGA_VERSION);
    printf("%s\n", rga_info);

    // rknn init
    rknn_context ctx = 0;
    assert(rknn_init(&ctx, model_path, 0, 0, NULL) == 0);

    // query model IO num
    rknn_input_output_num io_num;
    assert(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(rknn_input_output_num)) == 0 );

    // query model IO attribute
    rknn_tensor_attr in_attr[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++) {
        in_attr[i].index = i;
        in_attr[i].type = RKNN_TENSOR_UINT8;
        assert(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(in_attr[i]), sizeof(rknn_tensor_attr)) == 0);
        my_tools::dump_tensor_attr(&in_attr[i]);
    }
    rknn_tensor_attr out_attr[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++) {
        out_attr[i].index = i;
        assert(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(out_attr[i]), sizeof(rknn_tensor_attr)) == 0);
        my_tools::dump_tensor_attr(&out_attr[i]); 
    }


    // DMA
    int64_t preprocess_us = my_tools::getCurrentTimeUs();
    unsigned char *img_buf;
    int input_dma_fd = -1;
    int ret = 0;
    int h = 256, w = 256, img_format = RK_FORMAT_BGR_888;
    int input_buf_size = h * w * get_bpp_from_format(img_format);
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, input_buf_size, &input_dma_fd, (void **)&img_buf);
    assert(ret == 0);
    img_buf = my_tools::load_image(img_path, &in_attr[0]);
    assert(img_buf != NULL);
    rga_buffer_handle_t input_buffer_handle = importbuffer_fd(input_dma_fd, input_buf_size);
    assert(input_buffer_handle != 0);
    rga_buffer_t input_rga_buffer = wrapbuffer_handle(input_buffer_handle, w, h, img_format);
    rknn_tensor_mem *input_mem = rknn_create_mem_from_fd(ctx, input_dma_fd, img_buf, in_attr->size_with_stride, 0);
    assert(input_mem != NULL);
    for (int i = 0; i < io_num.n_input; i++) {
        ret = rknn_set_io_mem(ctx, &input_mem[i], &in_attr[i]);
        assert(ret == 0);
    }
    // DMA for output
    rknn_tensor_mem *output_mem[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++) {
        output_mem[i] = rknn_create_mem(ctx, out_attr[i].size_with_stride);
        ret = rknn_set_io_mem(ctx, output_mem[i], &out_attr[i]);
        assert(ret == 0);
    }
    preprocess_us = my_tools::getCurrentTimeUs() - preprocess_us;

    // RUN!
    int64_t inference_us = my_tools::getCurrentTimeUs();
    ret = rknn_run(ctx, NULL);
    inference_us = my_tools::getCurrentTimeUs() - inference_us;

    printf("Preprocess Time = %.2fms, FPS = %.2f\n", preprocess_us / 1000.f, 1000.f * 1000.f / preprocess_us);
    printf("Inference Time = %.2fms, FPS = %.2f\n", inference_us / 1000.f, 1000.f * 1000.f / inference_us);


    printf("rknn run status code: %d\n", ret);

    // post process

    // angle
    int index = 1;
    int zp = out_attr[index].zp;
    float scale = out_attr[index].scale;
    int8_t value = 0;
    for (int i = 0; i < out_attr[index].n_elems; i++) {
        value = ((int8_t *)output_mem[index]->virt_addr)[i];
        printf("%.4f ", (value - zp) / scale);
    }
    printf("\n");

    return 0;
}
