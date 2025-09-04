#include <stdio.h>
#include <assert.h>
// #include "rknn_api.h"
#include "tools.h"
#include "opencv2/opencv.hpp"

int main(int nargs, char **vargs) {
    assert(nargs == 3);
    char *model_path = vargs[1];
    char *image_path = vargs[2];

    int ret = 0;

    // rknn init
    rknn_context ctx = -1;
    assert(!rknn_init(&ctx, model_path, 0, 0, NULL));
    rknn_sdk_version sdk_ver;
    assert(!rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(rknn_sdk_version)));
    printf("RKNN SDK api version: %s, driver version: %s.\n", sdk_ver.api_version, sdk_ver.drv_version);
    rknn_input_output_num io_num;
    assert(!rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(rknn_input_output_num)));
    printf("model input num: %d, output num: %d.\n", io_num.n_input, io_num.n_output);
    rknn_tensor_attr in_attrs[io_num.n_input], out_attrs[io_num.n_output];
    printf("Input tensor attr:\n");
    for (int i = 0; i < io_num.n_input; i++) {
        in_attrs[i].index = i;
        // do not need to assign type and fmt manually, these are determined by model, and hardware.
        // in_attrs[i].fmt = RKNN_TENSOR_NHWC;
        // in_attrs[i].type = RKNN_TENSOR_UINT8;
        // printf("before query data type: %d\n", in_attrs[i].type == RKNN_TENSOR_UINT8);
        assert(!rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(rknn_tensor_attr)));
        // printf("after query data type: %d\n", in_attrs[i].type == RKNN_TENSOR_UINT8);
        tools::dump_tensor_attr(&in_attrs[i]);
    }
    printf("Output tensor attr:\n");
    for (int i = 0; i < io_num.n_output; i++) {
        out_attrs[i].index = i;
        // out_attrs[i].type = RKNN_TENSOR_UINT8;
        assert(!rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr)));
        tools::dump_tensor_attr(&out_attrs[i]);
    }
    // rknn init end

    int req_n = in_attrs[0].dims[0];
    int req_h = in_attrs[0].dims[1];
    int req_w = in_attrs[0].dims[2];
    int req_c = in_attrs[0].dims[3];

    // preprocess
    // resize
    cv::Mat image_original = cv::imread(image_path, cv::IMREAD_COLOR);
    assert(!image_original.empty());
    cv::Mat image_resized;
    cv::resize(image_original, image_resized, cv::Size(req_w, req_h));
    assert(!image_resized.empty());

    rknn_tensor_mem *input_mem[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++) {
        input_mem[i] = rknn_create_mem(ctx, in_attrs[i].size_with_stride);
        memcpy(input_mem[i]->virt_addr, image_resized.data, in_attrs[i].size_with_stride);
        assert(!rknn_set_io_mem(ctx, input_mem[i], &in_attrs[i]));
    }

    rknn_tensor_mem *output_mem[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++) {
        // for YOLO, the output must be NC1HWC2
        out_attrs[i].fmt = RKNN_TENSOR_NC1HWC2;
        output_mem[i] = rknn_create_mem(ctx, out_attrs[i].size_with_stride);
        assert(!rknn_set_io_mem(ctx, output_mem[i], &out_attrs[i]));
    }
    // preprocess end

    // inference
    assert(!rknn_run(ctx, NULL));
    // inference end

    // postprocess
    const int8_t *out_original = (int8_t *)output_mem[0]->virt_addr;
    int8_t *out_analyzed = (int8_t *) malloc(req_n * req_h * req_w * req_c * sizeof(int8_t));
    int dims[5] = {1, 16, 20, 20, 16};
    tools::NC1HWC2_i8_to_NCHW_i8(out_original, out_analyzed, dims);
    printf("hhh\n");
    return 0;
}