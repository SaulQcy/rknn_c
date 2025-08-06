#include <stdio.h>
#include <string.h>
#include "tools.h"
#include <cstdint>



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

int main(int argc, char *argv[])
{
  if (argc < 3)
    return -1;
  char *model_path = argv[1];
  char *img_path = argv[2];
  // rknn init
  rknn_context ctx = 0;
  if (rknn_init(&ctx, model_path, 0, 0, NULL) != 0)
    return -2;
  rknn_sdk_version sdk_v;
  if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_v, sizeof(sdk_v)) != 0)
    return -3;
  rknn_input_output_num io_num;
  if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != 0)
    return -4;
  printf("input num %d, output num %d. \n", io_num.n_input, io_num.n_output);
  printf("input tensor:\n");
  rknn_tensor_attr in_attr[io_num.n_input];
  in_attr[0].fmt = RKNN_TENSOR_NHWC;
  in_attr[0].type = RKNN_TENSOR_UINT8;
  for (int i = 0; i < io_num.n_input; i++)
  {
    in_attr[i].index = i;
    if (rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(in_attr[i]), sizeof(rknn_tensor_attr)) != 0)
      return -5;
    dump_tensor_attr(&(in_attr[i]));
  }
  printf("\n");

  printf("output tensor:\n");
  rknn_tensor_attr out_attr[io_num.n_output];
  out_attr[0].type = RKNN_TENSOR_INT8;
  // out_attr[0].type = RKNN_TENSOR_UINT8;

  for (int i = 0; i < io_num.n_output; i++)
  {
    out_attr[i].index = i;
    if (rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(out_attr[i]), sizeof(rknn_tensor_attr)) != 0)
      return -6;
    dump_tensor_attr(&(out_attr[i]));
  }
  printf("\n");

  unsigned char *input_data = NULL;
  input_data = tools::load_image(img_path, &(in_attr[0]));
  if (input_data == NULL)
    return -7;
  rknn_tensor_mem *input_mem[1];
  input_mem[0] = rknn_create_mem(ctx, in_attr[0].size_with_stride);

  int h = in_attr[0].dims[1];
  int w = in_attr[0].dims[2];
  int c = in_attr[0].dims[3];
  int stride = in_attr[0].w_stride;

  if (w == stride)
  {
    printf("width == stride\n");
    memcpy(input_mem[0]->virt_addr, input_data, h * w * c);
  }

  // check the values in board end.
  // for (int i = 0; i < 1000; ++i)
  // {
  //   printf("%d ", ((uint8_t *)input_mem[0]->virt_addr)[i]);
  //   if ((i + 1) % 20 == 0)
  //     printf("\n");
  // }

  rknn_tensor_mem *output_mem[io_num.n_output];
  for (int i = 0; i < io_num.n_output; i++)
    output_mem[i] = rknn_create_mem(ctx, out_attr[i].size_with_stride);

  // set input/output tensor memory
  if (rknn_set_io_mem(ctx, input_mem[0], &in_attr[0]) != 0)
    return -8;

  for (int i = 0; i < io_num.n_output; i++)
    if (rknn_set_io_mem(ctx, output_mem[i], &out_attr[i]) != 0)
      return -8;

  printf("\nrun\n");
  if (rknn_run(ctx, NULL) != 0)
    return -9;

  // postprocess

  // landmark
  float res = 0;
  for (int i = 0; i < 38; i++) {
    // uint8_t value = ((int8_t *)output_mem[0]->virt_addr)[i];
    int8_t value = ((int8_t *)output_mem[0]->virt_addr)[i];
    res = (value - out_attr[0].zp) * out_attr[0].scale;
    printf("index: %d\t zp: %d\t scale: %.4f\t raw: %d\t val: %.4f\n", i, out_attr[0].zp, out_attr[0].scale, value, res);
  }
  
  // headpose
  for (int i = 0; i < 3; i++) {
    int8_t value = ((int8_t *)output_mem[1]->virt_addr)[i];
    res = (value - out_attr[1].zp) * out_attr[1].scale;
    printf("%.2f %d\n", res, value);

  }

  // main  class
  for (int i = 0; i < 4; i++) {
    int8_t value = ((int8_t *)output_mem[2]->virt_addr)[i];
    res = (value - out_attr[2].zp) * out_attr[2].scale;
    printf("index: %d\t zp: %d\t scale: %.4f\t raw: %d\t val: %.4f\n", i, out_attr[2].zp, out_attr[2].scale, value, res);
  }

  return 0;
}