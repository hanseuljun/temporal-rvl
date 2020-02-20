#pragma once

#include "rvl.h"

namespace trvl
{
struct Pixel
{
public:
    Pixel() : value(0), invalid_count(0) {}
    short value;
    short invalid_count;
};

short abs_diff(short x, short y)
{
    if (x > y)
        return x - y;
    else
        return y - x;
}

void update_pixel(Pixel& pixel, short raw_value, short change_threshold, int invalidation_threshold) {
    if (pixel.value == 0) {
        if (raw_value > 0)
            pixel.value = raw_value;

        return;
    }


    // Reset the pixel if the depth value indicates the input was invalid two times in a row.
    if (raw_value == 0) {
        ++pixel.invalid_count;
        if (pixel.invalid_count >= invalidation_threshold) {
            pixel.value = 0;
            pixel.invalid_count = 0;
        }
        return;
    }
    pixel.invalid_count = 0;

    // Update pixel value when change is detected.
    if (abs_diff(pixel.value, raw_value) > change_threshold)
        pixel.value = raw_value;
}

class Encoder
{
public:
    Encoder(int frame_size, short change_threshold, int invalid_threshold)
        : pixels_(frame_size), change_threshold_(change_threshold), invalid_threshold_(invalid_threshold)
    {
    }

    std::vector<char> Encoder::encode(short* depth_buffer, bool keyframe)
    {
        auto frame_size = pixels_.size();
        if (keyframe) {
            for (int i = 0; i < frame_size; ++i) {
                pixels_[i].value = depth_buffer[i];
                // Not sure this is the best way to set invalid_count...
                pixels_[i].invalid_count = depth_buffer[i] == 0 ? 1 : 0;
            }

            return rvl::compress(depth_buffer, frame_size);
        }

        std::vector<short> pixel_diffs(frame_size);
        for (int i = 0; i < frame_size; ++i) {
            pixel_diffs[i] = pixels_[i].value;
            update_pixel(pixels_[i], depth_buffer[i], change_threshold_, invalid_threshold_);
            pixel_diffs[i] = pixels_[i].value - pixel_diffs[i];
        }

        return rvl::compress(pixel_diffs.data(), frame_size);
    }

private:
    std::vector<Pixel> pixels_;
    short change_threshold_;
    int invalid_threshold_;
};

class Decoder
{
public:
    Decoder(int frame_size)
        : prev_pixel_values_(frame_size, 0)
    {
    }

    std::vector<short> decode(char* trvl_frame, bool keyframe)
    {
        int frame_size = prev_pixel_values_.size();
        if (keyframe) {
            prev_pixel_values_ = rvl::decompress(trvl_frame, frame_size);
            return prev_pixel_values_;
        }

        auto pixel_diffs = rvl::decompress(trvl_frame, frame_size);
        for (int i = 0; i < frame_size; ++i)
            prev_pixel_values_[i] += pixel_diffs[i];

        return prev_pixel_values_;
    }

private:
    std::vector<short> prev_pixel_values_;
};
}