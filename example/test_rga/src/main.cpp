//
// Created by saul on 2025/8/7.
//
#include <assert.h>
#include <iostream>
#include <ostream>
#include <string>
#include "im2d_version.h"
#include "im2d_common.h"
#include "utils.h"

int main(int n_args, char **args) {
    assert(n_args == 2);
    char *img_path = args[1];

    // RGA information
    const char *rga_info = querystring(RGA_ALL);
    printf("%s\n", rga_info);



    return 0;
}
