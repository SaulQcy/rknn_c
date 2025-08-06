#include "postprocess.h"

std::vector<float> pfld_postprocess::extract_landmark(rknn_tensor_mem* output_mem, rknn_tensor_attr out_attr) {
    std::vector<float> ret;

    for (int i = 0; i < 38; i++) {
        // uint8_t value = ((int8_t *)output_mem[0]->virt_addr)[i];
        int8_t value = ((int8_t *)output_mem[0]->virt_addr)[i];
        ret.push_back((value - out_attr[0].zp) * out_attr[0].scale);
        // printf("index: %d\t zp: %d\t scale: %.4f\t raw: %d\t val: %.4f\n", i, out_attr[0].zp, out_attr[0].scale, value, res);
    }

    return ret;
}