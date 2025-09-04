#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/kernel.h>
#include <linux/gpio.h>

#define ADC_PATH "/sys/bus/iio/devices/iio:device0"

int main() {
    printf("hello adc\n");

    FILE *f_in0 = fopen(ADC_PATH "/in_voltage0_raw", "r");
    FILE *f_in1 = fopen(ADC_PATH "/in_voltage1_raw", "r");
    FILE *f_scale = fopen(ADC_PATH "/in_voltage_scale", "r");

    char buf[20];
    int in0, in1;
    float scale, in0_v, in1_v;

    while (1) {
        fseek(f_scale, 0, SEEK_SET);
        fseek(f_in0, 0, SEEK_SET);
        fseek(f_in1, 0, SEEK_SET);
        if (f_in0 != NULL && f_in1 != NULL && f_scale != NULL) {
            fgets(buf, sizeof(buf), f_in0);
            in0 = atoi(buf);

            fgets(buf, sizeof(buf), f_in1);
            in1 = atoi(buf);

            fgets(buf, sizeof(buf), f_scale);
            scale = strtof(buf, NULL);

            in0_v = (in0 * scale) / 1000.;
            in1_v = (in1 * scale) / 1000.;

            printf("in1: %.4fV, in2: %.4fV\n", in0_v, in1_v);

        } else break;
        sleep(1);
    } 

    return 0;
}