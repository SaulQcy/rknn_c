#include <stdio.h>
#include <rknn_api.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

static unsigned char *load_image(const char *image_path, rknn_tensor_attr *input_attr)
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

  int height = 0;
  int width = 0;
  int channel = 0;

  unsigned char *image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
  if (image_data == NULL)
  {
    printf("load image failed!\n");
    return NULL;
  }

  if (width != req_width || height != req_height)
  {
    unsigned char *image_resized = (unsigned char *)STBI_MALLOC(req_width * req_height * req_channel);
    if (!image_resized)
    {
      printf("malloc image failed!\n");
      STBI_FREE(image_data);
      return NULL;
    }
    if (stbir_resize_uint8(image_data, width, height, 0, image_resized, req_width, req_height, 0, channel) != 1)
    {
      printf("resize image failed!\n");
      STBI_FREE(image_data);
      return NULL;
    }
    STBI_FREE(image_data);
    image_data = image_resized;
  }

  return image_data;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
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

int main(int argc, char *argv[]) {
    if (argc < 3) return -1;
    char *model_path = argv[1];
    char *img_path = argv[2];
    // rknn init
    rknn_context ctx = 0;
    if (rknn_init(&ctx, model_path, 0, 0, NULL) != 0) return -2;
    rknn_sdk_version sdk_v;
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_v, sizeof(sdk_v)) != 0) return -3;
    rknn_input_output_num io_num;
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != 0) return -4;
    printf("input num %d, output num %d. \n", io_num.n_input, io_num.n_output);
    printf("input tensor:\n");
    rknn_tensor_attr in_attr[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++) {
        in_attr[i].index = i;
        if (rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(in_attr[i]), sizeof(rknn_tensor_attr)) != 0) return -5;
        dump_tensor_attr(&(in_attr[i]));
    }
    printf("\n");
    printf("output tensor:\n");
    rknn_tensor_attr out_attr[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++) {
        out_attr[i].index = i;
        if (rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(out_attr[i]), sizeof(rknn_tensor_attr)) != 0) return -6;
        dump_tensor_attr(&(out_attr[i]));
    }
    printf("\n");

    unsigned char *input_data = NULL;
    input_data = load_image(img_path, &(in_attr[0]));
    if (input_data == NULL) return -7;
    

    return 0;
}