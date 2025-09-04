// tools.h
#pragma once

#include "rknn_api.h"
#include <stdio.h>

class tools {
public:
    static unsigned char* load_image(const char *, rknn_tensor_attr *);
    static void dump_tensor_attr(rknn_tensor_attr *);
    static int64_t getCurrentTimeUs();
    static int NC1HWC2_i8_to_NHWC_i8(const int8_t* src, int8_t* dst, int* dims, int channel, int h, int w);
    static int NC1HWC2_i8_to_NCHW_i8(const int8_t* src, int8_t* dst, int* dims);
};