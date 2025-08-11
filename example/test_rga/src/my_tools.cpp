#include "my_tools.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>


void my_tools::dump_tensor_attr(rknn_tensor_attr *attr)
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

void read_image_RGBA(char *img_path) {
    int ret = 0;
    int src_w = 640, src_h = 640, src_format = RK_FORMAT_RGBA_8888;
    int src_buffer_size = src_w * src_h * get_bpp_from_format(src_format);
    char *src_buffer = (char *) malloc(src_buffer_size);
    FILE *file = fopen(img_path, "rb");
    if (!file) {
        fprintf(stderr, "Could not open %s\n", img_path);
        free(src_buffer);
        return;
    }
    fread(src_buffer, src_buffer_size, 1, file);
    fclose(file);

    draw_rgba(src_buffer, src_h, src_w);
}

void read_img_BGR(char *img_path) {
    const char *rga_info = querystring(RGA_VERSION);
    printf("%s\n", rga_info);

    int h = 640, w = 640, img_format = RK_FORMAT_BGR_888;
    int img_buf_size = h * w * get_bpp_from_format(img_format);
    FILE *file = fopen(img_path, "rb");
    if (!file) {
        printf("Could not open %s\n", img_path);
    }
    char *img_buf = (char *)malloc(img_buf_size);
    fread(img_buf, img_buf_size, 1, file);
    fclose(file);

    free(img_buf);
    printf("end\n");
}