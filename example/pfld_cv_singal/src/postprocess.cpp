#include "postprocess.h"

void pfld_results::print_results() {
    printf("landmark: \n");
    for (auto val : landmark) {
        printf("%.4f ", val);
    }
    printf("\n");

    printf("headpose:\n");
    for (auto val : headpose) {
        printf("%.4f ", val);
    }
    printf("\n");

    printf("main classes:\n");
    for (auto val : main_classes) {
        printf("%s ", val ? "True" : "False");
    }
    printf("\nover\n");
}


pfld_results pfld_postprocess::extract_landmark(rknn_tensor_mem **output_mem, rknn_tensor_attr *out_attr) {
    pfld_results ret;

    // landmark
    for (int i = 0; i < 38; i++) {
        // uint8_t value = ((int8_t *)output_mem[0]->virt_addr)[i];
        int8_t value = ((int8_t *)output_mem[0]->virt_addr)[i];
        ret.landmark.push_back((value - out_attr[0].zp) * out_attr[0].scale);
        // printf("index: %d\t zp: %d\t scale: %.4f\t raw: %d\t val: %.4f\n", i, out_attr[0].zp, out_attr[0].scale, value, res);
    }

    // headpose
    for (int i = 0; i < 3; i++) {
        int8_t value = ((int8_t *)output_mem[1]->virt_addr)[i];
        ret.headpose.push_back(
            (value - out_attr[1].zp) * out_attr[1].scale
        );
    }

    // main  class
    for (int i = 0; i < 4; i++) {
        int8_t value = ((int8_t *)output_mem[2]->virt_addr)[i];
        ret.main_classes.push_back(
            ((value - out_attr[2].zp) * out_attr[2].scale) >= 0
        );
    }

    return ret;
}