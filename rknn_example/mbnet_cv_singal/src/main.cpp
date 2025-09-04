#include <stdio.h>
#include <assert.h>
// #include "rknn_api.h"
#include "tools.h"
#include "opencv2/opencv.hpp"
#include "vector"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>

void print_topk(int8_t* data, int n, int zp, float scale, int k = 10) {
    std::vector<float> logits(n);
    for (int i = 0; i < n; i++) {
        logits[i] = (data[i] - zp) * scale;
    }

    // softmax (数值稳定版)
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> expv(n);
    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        expv[i] = std::exp(logits[i] - max_logit);
        sum_exp += expv[i];
    }
    std::vector<float> probs(n);
    for (int i = 0; i < n; i++) {
        probs[i] = expv[i] / sum_exp;
    }

    // 索引数组
    std::vector<int> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i;

    // 按概率排序
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    // 打印前 k
    for (int i = 0; i < k; i++) {
        int idx = indices[i];
        printf("Index: %d, Prob: %.4f, Logit: %.4f\n", 
               idx, probs[idx], logits[idx]);
    }
}


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
    // 使用方法
    int zp = out_attrs[0].zp;
    float scale = out_attrs[0].scale;
    int8_t* raw_data = (int8_t*)output_mem[0]->virt_addr;
    print_topk(raw_data, 1000, zp, scale, 10);
    // postprocess end

    return 0;
}

