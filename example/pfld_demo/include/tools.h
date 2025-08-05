// tools.h
#pragma once

#include "rknn_api.h"
#include <stdio.h>

class tools {
public:
    tools();
    ~tools();

    // 静态函数：加载并调整图像大小以匹配模型输入
    static unsigned char* load_image(const char* image_path, rknn_tensor_attr* input_attr);
};