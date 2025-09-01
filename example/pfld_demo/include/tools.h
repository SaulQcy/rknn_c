// tools.h
#pragma once

#include "rknn_api.h"
#include <stdio.h>

class tools {
public:
    static unsigned char* load_image(const char *, rknn_tensor_attr *);
    static void dump_tensor_attr(rknn_tensor_attr *);
    static int64_t getCurrentTimeUs();
};