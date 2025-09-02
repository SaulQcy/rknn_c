// tools.cpp
#include "tools.h"

// 只在一个 cpp 文件中定义实现
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize.h"
#include "sys/time.h"

void tools::dump_tensor_attr(rknn_tensor_attr *attr)
{
  char dims[128] = {0}; // to store the shape of input/output tensor shape, e.g., [1, 224, 224, 3] NHWC, [1, 1001]
  for (int i = 0; i < attr->n_dims; ++i)
  {
    int idx = strlen(dims); // the current length of dims
    sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
  }
  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, get_format_string(attr->fmt),
         get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


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


int64_t tools::getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

int tools::NC1HWC2_i8_to_NHWC_i8(const int8_t* src, int8_t* dst, int* dims, int channel, int h, int w)
{
  int batch  = dims[0];
  int C1     = dims[1];
  int C2     = dims[4];
  int hw_src = dims[2] * dims[3];
  int hw_dst = h * w;
  for (int i = 0; i < batch; i++) {
    const int8_t* src_b = src + i * C1 * hw_src * C2;
    int8_t*       dst_b = dst + i * channel * hw_dst;
    for (int cur_h = 0; cur_h < h; ++cur_h)
      for (int cur_w = 0; cur_w < w; ++cur_w) {
        for (int c = 0; c < channel; ++c) {
          int           plane  = c / C2;
          const int8_t* src_bc = plane * hw_src * C2 + src_b;
          int           offset = c % C2;
          int cur_hw                 = cur_h * w + cur_w;
          dst_b[cur_h * w * channel + cur_w* channel + c] = src_bc[C2 * cur_hw + offset] ;

        }
    }
  }
  return 0;
}



