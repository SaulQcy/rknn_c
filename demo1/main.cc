#include <stdio.h>
#include <string.h>
#include "rknn_api.h"
#include "demo1_utils.h"
#include "stdlib.h"
#include "fp16/Float16.h"

using namespace rknpu2;


int NC1HWC2_fp16_to_NCHW_fp32(const float16* src, float* dst, int* dims, int channel, int h, int w, int zp, float scale)
{
  int batch  = dims[0];
  int C1     = dims[1];
  int C2     = dims[4];
  int hw_src = dims[2] * dims[3];
  int hw_dst = h * w;
  for (int i = 0; i < batch; i++) {
    const float16* src_b = src + i * C1 * hw_src * C2;
    float*         dst_b = dst + i * channel * hw_dst;
    for (int c = 0; c < channel; ++c) {
      int            plane  = c / C2;
      const float16* src_bc = plane * hw_src * C2 + src_b;
      int            offset = c % C2;
      for (int cur_h = 0; cur_h < h; ++cur_h)
        for (int cur_w = 0; cur_w < w; ++cur_w) {
          int cur_hw                 = cur_h * w + cur_w;
          dst_b[c * hw_dst + cur_hw] = src_bc[C2 * cur_hw + offset]; // float16-->float
        }
    }
  }

  return 0;
}

int NC1HWC2_int8_to_NCHW_float(const int8_t *src, float *dst, int *dims, int channel, int h, int w, int zp, float scale)
{
  int batch = dims[0];
  int C1 = dims[1];
  int C2 = dims[4];
  int hw_src = dims[2] * dims[3];
  int hw_dst = h * w;
  for (int i = 0; i < batch; i++)
  {
    src = src + i * C1 * hw_src * C2;
    dst = dst + i * channel * hw_dst;
    for (int c = 0; c < channel; ++c)
    {
      int plane = c / C2;
      const int8_t *src_c = plane * hw_src * C2 + src;
      int offset = c % C2;
      for (int cur_h = 0; cur_h < h; ++cur_h)
        for (int cur_w = 0; cur_w < w; ++cur_w)
        {
          int cur_hw = cur_h * w + cur_w;
          dst[c * hw_dst + cur_h * w + cur_w] = (src_c[C2 * cur_hw + offset] - zp) * scale; // int8-->float
        }
    }
  }

  return 0;
}


int main(int argc, char* argv[])
{
    printf("print args %d\n", argc);
    for (int i = 1; i < argc; i++)
    {
        printf("%s\n", argv[i]);
    }
    char* img_path = argv[2];
    char* model_path = argv[1];
    
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model_path, 0, 0, NULL);
    if (ret < 0)
    {
        printf("rknn init fail, return is %d\n", ret);
        return -1;
    }
    printf("after init, the ctx is %d\n", ctx);

    // query the SDK version
    rknn_sdk_version sdk_version;
    ret = rknn_query(
        ctx, 
        RKNN_QUERY_SDK_VERSION, 
        &sdk_version, 
        sizeof(sdk_version)
    );
    if (ret != RKNN_SUCC)
    {
        printf("rknn query fail, return is %d\n", ret);
        return -1;
    }
    printf("rknn info, rknnrt version: %s, driver vresion: %s\n", sdk_version.api_version, sdk_version.drv_version);
    
    // query the RKNN IO path
    rknn_input_output_num io_num;
    ret = rknn_query(
        ctx,
        RKNN_QUERY_IN_OUT_NUM,
        &io_num,
        sizeof(io_num)
    );
    if (ret != 0)
    {
        printf("rknn query faild! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // query the input attribute, e.g., shape, scale, zero-point.
    printf("input tensor: \n");
    rknn_tensor_attr input_attr[io_num.n_input];
    memset(input_attr, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attr[i].index = i;
        ret = rknn_query(
            ctx,
            RKNN_QUERY_INPUT_ATTR,
            &(input_attr[i]),
            sizeof(rknn_tensor_attr)
        );
        if (ret != 0)
        {
            printf("rknn query input fail: %d\n", ret);
            return ret;
        }
        Tools::dump_tensor_attr(&input_attr[i]);
    }

    // query the output attributes, e.g., shape, scale, zp.
    printf("output tensor: \n");
    rknn_tensor_attr output_attr[io_num.n_output];
    memset(output_attr, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attr[i].index = i;
        ret = rknn_query(
            ctx,
            RKNN_QUERY_OUTPUT_ATTR,
            &(output_attr[i]),
            sizeof(rknn_tensor_attr)
        );
        if (ret != 0)
        {
            printf("rknn output attr query fail: %d\n", int(ret));
            return ret;
        }
        Tools::dump_tensor_attr(&output_attr[i]);
    }

    unsigned char *input_data = NULL;
    rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
    rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;
    input_data = Tools::load_image(img_path, &input_attr[0]);
    if (!input_data)
        return -1;
    
    rknn_tensor_mem *input_mems[1];
    input_attr[0].type = input_type;
    input_attr[0].fmt = input_layout;
    input_mems[0] = rknn_create_mem(ctx, input_attr[0].size_with_stride);
    int width = input_attr[0].dims[2];
    int stride = input_attr[0].w_stride;
    if (width == stride)
    {
        memcpy(input_mems[0]->virt_addr, input_data, width * input_attr[0].dims[1] * input_attr[0].dims[3]);
    }
    else
    {
        int height = input_attr[0].dims[1];
        int channel = input_attr[0].dims[3];
        // copy from src to dst with stride
        uint8_t *src_ptr = input_data;
        uint8_t *dst_ptr = (uint8_t *)input_mems[0]->virt_addr;
        // width-channel elements
        int src_wc_elems = width * channel;
        int dst_wc_elems = stride * channel;
        for (int h = 0; h < height; ++h)
        {
            memcpy(dst_ptr, src_ptr, src_wc_elems);
            src_ptr += src_wc_elems;
            dst_ptr += dst_wc_elems;
        }
    }
    // Create output tensor memory
    rknn_tensor_mem *output_mems[io_num.n_output];
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        output_mems[i] = rknn_create_mem(ctx, output_attr[i].n_elems * sizeof(float));
    }

    // Set input tensor memory
    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attr[0]);
    if (ret < 0)
    {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
    }

    // Set output tensor memory
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        // set output memory and attribute
        output_attr[i].type = RKNN_TENSOR_FLOAT32;
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attr[i]);
        if (ret < 0)
        {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    int loop_count = 1;

    // Run
    printf("Begin perf ...\n");
    for (int i = 0; i < loop_count; ++i)
    {
        int64_t start_us = Tools::getCurrentTimeUs();
        ret = rknn_run(ctx, NULL);
        int64_t elapse_us = Tools::getCurrentTimeUs() - start_us;
        if (ret < 0)
        {
            printf("rknn run error %d\n", ret);
            return -1;
        }
        printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", i, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
    }

  printf("output origin tensors:\n");
  rknn_tensor_attr orig_output_attrs[io_num.n_output];
  memset(orig_output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_output; i++)
  {
    orig_output_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(orig_output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    Tools::dump_tensor_attr(&orig_output_attrs[i]);
  }

  float *output_mems_nchw[io_num.n_output];
  for (uint32_t i = 0; i < io_num.n_output; ++i)
  {
    int size = orig_output_attrs[i].size_with_stride * sizeof(float);
    output_mems_nchw[i] = (float *)malloc(size);
  }

  for (uint32_t i = 0; i < io_num.n_output; i++)
  {
    if (output_attr[i].fmt == RKNN_TENSOR_NC1HWC2)
    {
      int channel = orig_output_attrs[i].dims[1];
      int h = orig_output_attrs[i].n_dims > 2 ? orig_output_attrs[i].dims[2] : 1;
      int w = orig_output_attrs[i].n_dims > 3 ? orig_output_attrs[i].dims[3] : 1;
      int zp = output_attr[i].zp;
      float scale = output_attr[i].scale;

      if (orig_output_attrs[i].type == RKNN_TENSOR_INT8) {
        NC1HWC2_int8_to_NCHW_float((int8_t *)output_mems[i]->virt_addr, (float *)output_mems_nchw[i], (int *)output_attr[i].dims,
                                 channel, h, w, zp, scale);
      } else {
        printf("dtype: %s cannot convert!", get_type_string(orig_output_attrs[i].type));
      }
    }
    else
    {
      int8_t *src = (int8_t *)output_mems[i]->virt_addr;
      float *dst = output_mems_nchw[i];
      for (int index = 0; index < output_attr[i].n_elems; index++)
      {
        dst[index] = (src[index] - output_attr[i].zp) * output_attr[i].scale;
      }
    }
  }

    // Get top 5
    uint32_t topNum = 5;
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        uint32_t MaxClass[topNum];
        float fMaxProb[topNum];
        uint32_t sz = output_attr[i].n_elems;
        int top_count = sz > topNum ? topNum : sz;
        float *buffer = (float *)output_mems[i]->virt_addr;
        Tools::rknn_GetTopN(buffer, fMaxProb, MaxClass, sz, topNum);
        printf("---- Top%d ----\n", top_count);
        for (int j = 0; j < top_count; j++)
        {
            printf("%8.6f - %d\n", fMaxProb[j], MaxClass[j]);
        }
    }

    // Destroy rknn memory
    rknn_destroy_mem(ctx, input_mems[0]);
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        rknn_destroy_mem(ctx, output_mems[i]);
    }

    // destroy
    rknn_destroy(ctx);

    free(input_data);

    return 0;
}