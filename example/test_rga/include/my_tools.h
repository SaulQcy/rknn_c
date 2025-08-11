#include "rknn_api.h"
#include "rga.h"
#include "RgaUtils.h"
#include "utils.h"
#include "im2d_common.h"
#include "im2d_version.h"

class my_tools {
    public:
        static void dump_tensor_attr(rknn_tensor_attr *);
        static unsigned char *load_image(const char *, int, int, int);
};