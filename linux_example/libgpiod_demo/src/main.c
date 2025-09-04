#include <gpiod.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    struct gpiod_chip *chip;
    struct gpiod_chip_info *info;

    chip = gpiod_chip_open("/dev/gpiochip0");
    if (!chip) { perror("open chip"); return -1; }

    info = gpiod_chip_get_info(chip);
    printf("%s\n", gpiod_chip_info_get_name(info));
    printf("%s\n", gpiod_chip_get_path(chip));
    printf("%d\n", gpiod_chip_info_get_num_lines(info));
    return 0;
}
