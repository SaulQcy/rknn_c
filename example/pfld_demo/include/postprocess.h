#pragma once
#include "rknn_api.h"
#include <vector>


class pfld_postprocess
{
public:
    std::vector<float> extract_landmark(rknn_tensor_mem* output_mem, rknn_tensor_attr out_attr);

};

