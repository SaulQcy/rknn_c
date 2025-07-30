#include "rknn_api.h"

class Tools
{
    public:
        static void dump_tensor_attr(rknn_tensor_attr *attr);
        static unsigned char *load_image(const char *image_path, rknn_tensor_attr *input_attr);
        static int64_t getCurrentTimeUs();
        static int rknn_GetTopN(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum);
};