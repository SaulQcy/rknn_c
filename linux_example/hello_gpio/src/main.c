#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

int export_gpio(int gpio_pin) {
    assert(gpio_pin >=0 && gpio_pin <= 160); // the max gpio pin of RV1106 chip.
    FILE *f = fopen("/sys/class/gpio/export", "w");
    if (f == NULL) {
        printf("export file does not exist.");
        return -1;
    }
    fprintf(f, "%d", gpio_pin);
    fclose(f);
    return 0;
}

int unexport_gpio(int gpio_pin) {
    assert(gpio_pin >= 0 && gpio_pin <= 160);
    FILE *f = fopen("/sys/class/gpio/unexport", "w");
    if (f == NULL) {
        printf("export file does not exist.");
        return -1;
    }
    fprintf(f, "%d", gpio_pin);
    fclose(f);
    return 0;
}

int control_gpio(int gpio_pin, char* direction, int level) {
    assert(gpio_pin >= 0 && gpio_pin <= 160);
    assert(strcmp(direction, "in") == 0 || strcmp(direction, "out") == 0);
    assert(level == 0 || level == 1);

    char f_path[60];
    FILE *f;

    // 设置方向
    snprintf(f_path, sizeof(f_path), "/sys/class/gpio/gpio%d/direction", gpio_pin);
    printf("write direction to %s.\n", f_path);
    f = fopen(f_path, "w");
    if (!f) {
        printf("direction file of gpio%d does not exist.\n", gpio_pin);
        return -1;
    }
    fprintf(f, "%s", direction);
    fclose(f);

    // 如果是输出模式，写电平
    if (strcmp(direction, "out") == 0) {
        snprintf(f_path, sizeof(f_path), "/sys/class/gpio/gpio%d/value", gpio_pin);
        printf("write level %d to %s.\n", level, f_path);
        f = fopen(f_path, "w");
        if (!f) {
            printf("value file of gpio%d does not exist.\n", gpio_pin);
            return -1;
        }
        fprintf(f, "%d", level);
        fclose(f);

        // 打印确认
        char cmd[60];
        snprintf(cmd, sizeof(cmd), "cat /sys/class/gpio/gpio%d/value", gpio_pin);
        system(cmd);
    }

    return 0;
}


int main() {
    int gpio_pin;
    printf("input the gpio pin: ");
    scanf("%d", &gpio_pin);
    if (export_gpio(gpio_pin) != 0) return -1;
    // in direction does not allow write.
    control_gpio(gpio_pin, "out", 1);
    return 0;
}