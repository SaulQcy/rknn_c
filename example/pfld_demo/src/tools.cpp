// tools.cpp
#include "tools.h"

// 只在一个 cpp 文件中定义实现
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize.h"

tools::tools()
{
    // 构造函数可以留空，或初始化资源
}

tools::~tools()
{
    // 析构函数
}

// 注意：这里不要加 'static'
unsigned char* tools::load_image(const char* image_path, rknn_tensor_attr* input_attr)
{
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;

    switch (input_attr->fmt)
    {
    case RKNN_TENSOR_NHWC:
        req_height = input_attr->dims[1];
        req_width = input_attr->dims[2];
        req_channel = input_attr->dims[3];
        break;
    case RKNN_TENSOR_NCHW:
        req_height = input_attr->dims[2];
        req_width = input_attr->dims[3];
        req_channel = input_attr->dims[1];
        break;
    default:
        printf("meet unsupported layout\n");
        return NULL;
    }

    int width = 0;
    int height = 0;
    int channel = 0;

    unsigned char* image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("load image failed: %s\n", image_path);
        return NULL;
    }

    printf("Loaded image: %s, original size: %d x %d x %d\n", image_path, width, height, channel);

    // 如果尺寸不匹配，则进行 resize
    if (width != req_width || height != req_height)
    {
        unsigned char* image_resized = (unsigned char*)STBI_MALLOC(req_width * req_height * req_channel);
        if (!image_resized)
        {
            printf("malloc image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }

        if (stbir_resize_uint8(image_data, width, height, 0,
                               image_resized, req_width, req_height, 0, req_channel) != 1)
        {
            printf("resize image failed!\n");
            STBI_FREE(image_data);
            STBI_FREE(image_resized);
            return NULL;
        }

        STBI_FREE(image_data);  // 释放原图
        image_data = image_resized;

        printf("Resized image to: %d x %d x %d\n", req_width, req_height, req_channel);
    }

    return image_data;
}