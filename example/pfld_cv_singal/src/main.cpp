#include <stdio.h>
#include <string.h>
#include "tools.h"
#include <cstdint>
#include "postprocess.h"
#include "opencv2/opencv.hpp"



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
    tools::dump_tensor_attr(&(in_attr[i]));
  }
  printf("\n");

  printf("output tensor:\n");
  rknn_tensor_attr out_attr[io_num.n_output];
  out_attr[0].type = RKNN_TENSOR_INT8;

  for (int i = 0; i < io_num.n_output; i++)
  {
    out_attr[i].index = i;
    if (rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(out_attr[i]), sizeof(rknn_tensor_attr)) != 0)
      return -6;
    tools::dump_tensor_attr(&(out_attr[i]));
  }
  printf("\n");

  int64_t preprocess_us = tools::getCurrentTimeUs();
  // use opencv to load and resize image
  cv::Mat src_img = cv::imread(img_path, cv::IMREAD_COLOR);
  assert(!src_img.empty());
  cv::Mat resized;
  cv::resize(src_img, resized, cv::Size(256, 256));


  rknn_tensor_mem *input_mem[1];
  input_mem[0] = rknn_create_mem(ctx, in_attr[0].size_with_stride);

  int h = in_attr[0].dims[1];
  int w = in_attr[0].dims[2];
  int c = in_attr[0].dims[3];
  int stride = in_attr[0].w_stride;

  if (w == stride)
  {
    printf("width == stride\n");
    memcpy(input_mem[0]->virt_addr, resized.data, h * w * c);
  }
  
  rknn_tensor_mem *output_mem[io_num.n_output];
  for (int i = 0; i < io_num.n_output; i++)
    output_mem[i] = rknn_create_mem(ctx, out_attr[i].size_with_stride);

  // set input/output tensor memory
  if (rknn_set_io_mem(ctx, input_mem[0], &in_attr[0]) != 0)
    return -8;

  for (int i = 0; i < io_num.n_output; i++)
    if (rknn_set_io_mem(ctx, output_mem[i], &out_attr[i]) != 0)
      return -8;
  preprocess_us = tools::getCurrentTimeUs() - preprocess_us;

  printf("\nrun\n");
  int ret = 0;
  int64_t inference_us = tools::getCurrentTimeUs();
  ret = rknn_run(ctx, NULL);
  inference_us = tools::getCurrentTimeUs() - inference_us;


  printf("Preprocess Time = %.2fms, FPS = %.2f\n", preprocess_us / 1000.f, 1000.f * 1000.f / preprocess_us);
  printf("Inference Time = %.2fms, FPS = %.2f\n", inference_us / 1000.f, 1000.f * 1000.f / inference_us);

  // postprocess
  pfld_results post_proxy = pfld_postprocess::extract_landmark(output_mem, out_attr);
  post_proxy.print_results();

  return 0;
}