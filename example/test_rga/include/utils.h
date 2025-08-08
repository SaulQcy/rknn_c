#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
#include "rknn_api.h"

class utils {
    public:
        static unsigned char* load_img(const char *, rknn_tensor_attr *);
        static void dump_tensor_attr(rknn_tensor_attr *);
};