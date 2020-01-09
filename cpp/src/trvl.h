#pragma once

#include "rvl.h"

namespace trvl
{
class Pixel
{
public:
    Pixel() : value_(0), invalid_count_(0) {}

    short value() { return value_; }
    void set_value(int value) { value_ = value; }
    int invalid_count() { return invalid_count_; }
    void set_invalid_count(int invalid_count) { invalid_count_ = invalid_count; }

private:
    short value_;
    int invalid_count_;
};

void update_pixel(Pixel& pixel, short raw_value, short change_threshold, int invalidation_threshold) {
    if (pixel.value() == 0) {
        if (raw_value > 0)
            pixel.set_value(raw_value);

        return;
    }

    // Reset the pixel if the depth value indicates the input was invalid two times in a row.
    if (raw_value == 0) {
        pixel.set_invalid_count(pixel.invalid_count() + 1);
        if (pixel.invalid_count() >= invalidation_threshold) {
            pixel.set_value(0);
            pixel.set_invalid_count(0);
        }
        return;
    }
    pixel.set_invalid_count(0);

    short diff = pixel.value() - raw_value;
    if (diff < 0)
        diff = -diff;

    // Update pixel value when change is detected.
    if (diff > change_threshold)
        pixel.set_value(raw_value);
}
}