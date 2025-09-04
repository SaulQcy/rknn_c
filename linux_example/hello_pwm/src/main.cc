#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#define PWM_PATH "/sys/class/pwm/pwmchip11"
#define PERIOD_NS 1000000
#define MIN_DUTY_CYCLE_NS 0
#define MAX_DUTY_CYCLE_NS 1000000

int main() {
    printf("hello pwm\n");

    // export 0 to user space
    FILE *f_export = fopen(PWM_PATH "/export", "w");
    if (f_export == NULL) {
        perror("export file does not exist.\n");
        return -1;
    }
    fprintf(f_export, "%d", 0);
    fclose(f_export);

    // set the period T.
    FILE *f_period = fopen(PWM_PATH "/pwm0/period", "w");
    if (f_period == NULL) {
        perror("period file does not exist.\n");
        return -1;
    }
    fprintf(f_period, "%d", PERIOD_NS);
    fclose(f_period);

    // enable
    FILE *f_enale = fopen(PWM_PATH "/pwm0/enable", "w");
    if (f_enale == NULL) {
        perror("enable file does not exist.\n");
        return -1;
    }
    fprintf(f_enale, "%d", 1);
    fclose(f_enale);

    int direction = 1;
    int duty_cycle_us = 0;
    while (1) {
        duty_cycle_us += 1e4 * direction;
        if (duty_cycle_us == MAX_DUTY_CYCLE_NS)
            direction = -1;
        else if (duty_cycle_us == MIN_DUTY_CYCLE_NS)
            direction = 1;
        
        FILE *f_duty = fopen(PWM_PATH "/pwm0/duty_cycle", "w");
        if (f_duty == NULL) {
            perror("duty cycle file does not exist.\n");
            return -1;
        }
        fprintf(f_duty, "%d", duty_cycle_us);
        fclose(f_duty);

        usleep(5e4);
    }

    FILE *f_unexport = fopen(PWM_PATH "/unexport", "w");
    if (f_unexport == NULL) {
        perror("unexport file foes not exist.\n");
        return -1;
    }
    fprintf(f_unexport, "%d", 0);
    fclose(f_unexport);

    return 0;
}