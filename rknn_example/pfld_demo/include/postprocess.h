#pragma once
#include "rknn_api.h"
#include <vector>
#include <stdio.h>

class pfld_results {
    public:
        std::vector<float> landmark;
        std::vector<float> headpose;
        std::vector<bool> main_classes;

        void print_results();
};

class pfld_postprocess {
public:
    static pfld_results extract_landmark(rknn_tensor_mem **, rknn_tensor_attr *);

};


