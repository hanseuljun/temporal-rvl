#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "rvl.h"

// Helper function that converts 1-bit buffers into OpenCV Mats.
cv::Mat create_bool_mat(int width, int height, std::vector<bool>& bool_buffer)
{
    int frame_size = width * height;
    std::vector<char> char_frame(frame_size);
    std::vector<char> chroma_frame(frame_size);

    for (int i = 0; i < frame_size; ++i) {
        char_frame[i] = bool_buffer[i] * 255;
        chroma_frame[i] = 128;
    }

    cv::Mat y_channel(height, width, CV_8UC1, char_frame.data());
    cv::Mat cr_channel(height, width, CV_8UC1, chroma_frame.data());
    cv::Mat cb_channel(height, width, CV_8UC1, chroma_frame.data());

    std::vector<cv::Mat> y_cr_cb_channels;
    y_cr_cb_channels.push_back(y_channel);
    y_cr_cb_channels.push_back(cr_channel);
    y_cr_cb_channels.push_back(cb_channel);

    cv::Mat y_cr_cb_frame;
    cv::merge(y_cr_cb_channels, y_cr_cb_frame);

    cv::Mat bgr_frame = y_cr_cb_frame.clone();
    cvtColor(y_cr_cb_frame, bgr_frame, CV_YCrCb2BGR);
    return bgr_frame;
}

// Helper function that converts 16-bit buffers into OpenCV Mats.
cv::Mat create_depth_mat(int width, int height, const short* depth_buffer)
{
    int frame_size = width * height;
    std::vector<char> reduced_depth_frame(frame_size);
    std::vector<char> chroma_frame(frame_size);

    for (int i = 0; i < frame_size; ++i) {
        reduced_depth_frame[i] = depth_buffer[i] / 32;
        chroma_frame[i] = 128;
    }

    cv::Mat y_channel(height, width, CV_8UC1, reduced_depth_frame.data());
    cv::Mat cr_channel(height, width, CV_8UC1, chroma_frame.data());
    cv::Mat cb_channel(height, width, CV_8UC1, chroma_frame.data());

    std::vector<cv::Mat> y_cr_cb_channels;
    y_cr_cb_channels.push_back(y_channel);
    y_cr_cb_channels.push_back(cr_channel);
    y_cr_cb_channels.push_back(cb_channel);

    cv::Mat y_cr_cb_frame;
    cv::merge(y_cr_cb_channels, y_cr_cb_frame);

    cv::Mat bgr_frame = y_cr_cb_frame.clone();
    cvtColor(y_cr_cb_frame, bgr_frame, CV_YCrCb2BGR);
    return bgr_frame;
}

// A helper function that converts a AVFrame into a std::vector.
std::vector<uint8_t> convert_picture_plane_to_bytes(uint8_t* data, int line_size, int width, int height)
{
    std::vector<uint8_t> bytes(width * height);
    for (int i = 0; i < height; ++i)
        memcpy(bytes.data() + i * width, data + i * line_size, width);

    return bytes;
}

float max(std::vector<short> v1) {
    return *max_element(v1.begin(), v1.end());
}

float mse(std::vector<short> true_values, std::vector<short> encoded_values)
{
    assert(true_values.size() == encoded_values.size());

    int sum = 0;
    int count = 0;
    for (int i = 0; i < true_values.size(); ++i) {
        if (true_values[i] == 0)
            continue;
        short error = true_values[i] - encoded_values[i];
        sum += error * error;
        ++count;
    }
    return sum / (float)count;
}

class Clock
{
public:
    Clock() : time_point_(std::chrono::high_resolution_clock::now()) {}
    float milliseconds()
    {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - time_point_).count() / 1000.0f;
    }

private:
    std::chrono::steady_clock::time_point time_point_;
};

void run_rvl(std::ifstream& input, int width, int height)
{
    int frame_size = width * height;
    int depth_buffer_size = frame_size * sizeof(short);
    std::vector<short> depth_buffer(frame_size);

    int compressed_size_sum = 0;
    float time_sum = 0.0f;
    int frame_count = 0;
    while (!input.eof()) {
        input.read(reinterpret_cast<char*>(depth_buffer.data()), depth_buffer_size);

        Clock clock;
        auto rvl_frame = rvl::compress(depth_buffer.data(), frame_size);
        auto depth_image = rvl::decompress(rvl_frame.data(), frame_size);
        time_sum += clock.milliseconds();

        // RVL is a lossless algorithm.
        if (depth_buffer != depth_image) {
            std::cout << "Found compression loss..." << std::endl;
            return;
        }

        auto depth_mat = create_depth_mat(width, height, depth_image.data());

        cv::imshow("Depth", depth_mat);
        if (cv::waitKey(1) >= 0)
            break;

        compressed_size_sum += rvl_frame.size();
        ++frame_count;
    }

    float average_frame_size_ratio = compressed_size_sum / (float) (depth_buffer_size * frame_count);
    float average_time = time_sum / frame_count;
    std::cout << "Type: RVL" << std::endl
              << "average time: " << average_time << " ms" << std::endl
              << "average frame size ratio: " << average_frame_size_ratio << std::endl;
}

class TrvlPixel
{
public:
    TrvlPixel(int invalidation_threshold, short change_threshold)
        : invalidation_threshold_(invalidation_threshold), change_threshold_(change_threshold),
          value_(0), zero_count_(0),
          validated_(false), invalidated_(false), changed_(false)
    {
    }

    void update(short value)
    {
        validated_ = false;
        invalidated_ = false;
        changed_ = false;

        if (value_ == 0) {
            if (value > 0) {
                value_ = value;
                validated_ = true;
            }
            return;
        }

        // Reset the pixel if the depth value indicates the input was invalid two times in a row.
        if (value == 0) {
            ++zero_count_;
            if (zero_count_ >= invalidation_threshold_) {
                value_ = 0;
                zero_count_ = 0;
                invalidated_ = true;
            }
            return;
        }
        zero_count_ = 0;

        // Reset if the value is too different from value_;
        short diff = value_ - value;
        if (diff < 0)
            diff = -diff;

        if (diff > change_threshold_) {
            value_ = value;
            changed_ = true;
        }
    }

    short value() { return value_; }
    bool validated() { return validated_; }
    bool invalidated() { return invalidated_; }
    bool changed() { return changed_; }

private:
    int invalidation_threshold_;
    short change_threshold_;
    short value_;
    int zero_count_;
    bool validated_;
    bool invalidated_;
    bool changed_;
};

void run_temporal_rvl(std::ifstream& input, int width, int height, int invalidation_threshold, short change_threshold)
{
    int frame_size = width * height;
    int depth_buffer_size = frame_size * sizeof(short);
    std::vector<short> depth_buffer(frame_size);
    //std::vector<short> filtered_depth_buffer(frame_size);
    std::vector<TrvlPixel> pixels(frame_size, TrvlPixel(invalidation_threshold, change_threshold));
    std::vector<short> pixel_values(frame_size);
    std::vector<short> pixel_diffs(frame_size);
    std::vector<char> rvl_frame;
    std::vector<short> depth_image;

    int compressed_size_sum = 0;
    float time_sum = 0.0f;
    int frame_count = 0;
    float psnr_sum = 0.0f;
    float validation_ratio_sum = 0.0f;
    float invalidation_ratio_sum = 0.0f;
    float change_ratio_sum = 0.0f;

    while (!input.eof()) {
        input.read(reinterpret_cast<char*>(depth_buffer.data()), depth_buffer_size);
        Clock clock;
        for (int i = 0; i < frame_size; ++i) {
            pixels[i].update(depth_buffer[i]);
        }

        // Tried bilateral filtering. It produces terrible results.
        //bilateral_filter(depth_buffer.data(), filtered_depth_buffer.data(), width, height);
        //for (int i = 0; i < frame_size; ++i) {
        //    pixels[i].update(filtered_depth_buffer[i]);
        //}


        if (frame_count == 0) {
            for (int i = 0; i < frame_size; ++i) {
                pixel_values[i] = pixels[i].value();
            }

            // Just compress then decompress.
            rvl_frame = rvl::compress(pixel_values.data(), frame_size);
            depth_image = rvl::decompress(rvl_frame.data(), frame_size);
            time_sum += clock.milliseconds();
        } else {
            // Calculate value_diffs with the existing previous_values.
            for (int i = 0; i < frame_size; ++i) {
                short value = pixels[i].value();
                pixel_diffs[i] = value - pixel_values[i];
                pixel_values[i] = value;
            }

            // Compress and decompress pixel_diffs.
            rvl_frame = rvl::compress(pixel_diffs.data(), frame_size);
            auto diff_frame = rvl::decompress(rvl_frame.data(), frame_size);
            // Add the decompressed diffs to depth_image with the previous values.
            for (int i = 0; i < frame_size; ++i) {
                depth_image[i] += diff_frame[i];
            }
            time_sum += clock.milliseconds();

            // The code below here is only for visualization.
            std::vector<bool> diff_map(frame_size);
            std::vector<bool> validation_map(frame_size);
            std::vector<bool> invalidation_map(frame_size);
            std::vector<bool> change_map(frame_size);
            
            int validation_sum = 0;
            int invalidation_sum = 0;
            int change_sum = 0;
            
            for (int i = 0; i < frame_size; ++i) {
                diff_map[i] = pixel_diffs[i] != 0;
                validation_map[i] = pixels[i].validated();
                invalidation_map[i] = pixels[i].invalidated();
                change_map[i] = pixels[i].changed();

                if (validation_map[i])
                    ++validation_sum;
                if (invalidation_map[i])
                    ++invalidation_sum;
                if (change_map[i])
                    ++change_sum;
            }

            cv::imshow("Diff", create_bool_mat(width, height, diff_map));
            cv::imshow("Validation", create_bool_mat(width, height, validation_map));
            cv::imshow("Invalidation", create_bool_mat(width, height, invalidation_map));
            cv::imshow("Change", create_bool_mat(width, height, change_map));
            if (cv::waitKey(1) >= 0)
                break;

            validation_ratio_sum += validation_sum / (float)frame_size;
            invalidation_ratio_sum += invalidation_sum / (float)frame_size;
            change_ratio_sum += change_sum / (float)frame_size;
        }

        auto depth_mat = create_depth_mat(width, height, depth_image.data());

        cv::imshow("Depth", depth_mat);
        if (cv::waitKey(1) >= 0)
            break;

        compressed_size_sum += rvl_frame.size();
        ++frame_count;
        float mse_value = mse(depth_buffer, depth_image);
        if(mse_value > 0)
            psnr_sum += max(depth_buffer) / sqrt(mse_value);
    }

    float average_frame_size_ratio = compressed_size_sum / (float) (depth_buffer_size * frame_count);
    float average_time = time_sum / frame_count;
    float average_psnr = psnr_sum / frame_count;
    float average_validation_ratio = validation_ratio_sum / frame_count;
    float average_invalidation_ratio = invalidation_ratio_sum / frame_count;
    float average_change_ratio = change_ratio_sum / frame_count;
    std::cout << "Type: Temporal RVL 1" << std::endl
              << "average frame size ratio: " << average_frame_size_ratio << std::endl
              << "average time: " << average_time << " ms" << std::endl
              << "average PSNR: " << average_psnr << std::endl
              << "average validation ratio: " << average_validation_ratio << std::endl
              << "average invalidation ratio: " << average_invalidation_ratio << std::endl
              << "average change ratio: " << average_change_ratio << std::endl;
}

void run()
{
    std::cout << "Enter filename: ";

    std::string filename;
    std::cin >> filename;
    
    std::string file_path = "../../../data/" + filename;

    std::ifstream input(file_path, std::ios::binary);

    if (input.fail()) {
        std::cout << "Failed to open " << filename << std::endl;
        return;
    }

    int width;
    int height;
    int byte_size;
    input.read(reinterpret_cast<char*>(&width), sizeof(width));
    input.read(reinterpret_cast<char*>(&height), sizeof(height));
    input.read(reinterpret_cast<char*>(&byte_size), sizeof(byte_size));
    assert(byte_size == sizeof(short));

    std::cout << "Enter compression type (1: RVL, 2: VP8, 3: TRVL): ";
    std::string compression_type;
    std::cin >> compression_type;

    if (compression_type == "rvl") {
        run_rvl(input, width, height);
    } else if (compression_type == "trvl") {
        int invalidation_threshold;
        short change_threshold;
        std::cout << "Enter TRVL invalidation threshold: ";
        std::cin >> invalidation_threshold;
        std::cout << "Enter TRVL change threshold in mm: ";
        std::cin >> change_threshold;
        run_temporal_rvl(input, width, height, invalidation_threshold, change_threshold);
    } else {
        std::cout << "Invalid compression type..." << std::endl;
    }

    input.close();
}

void main()
{
    while(true)
        run();
}