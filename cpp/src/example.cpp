#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "trvl.h"

class InputFile
{
public:
    InputFile(std::string filename, std::ifstream&& input_stream, int width, int height)
        : filename_(filename), input_stream_(std::move(input_stream)), width_(width), height_(height) {}

    std::string filename() { return filename_; }
    std::ifstream& input_stream() { return input_stream_; }
    int width() { return width_; }
    int height() { return height_; }

private:
    std::string filename_;
    std::ifstream input_stream_;
    int width_;
    int height_;
};

std::vector<std::string> get_filenames_from_folder_path(std::string folder_path)
{
    std::vector<std::string> filenames;
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        std::string filename = entry.path().filename().string();
        if (filename == ".gitignore")
            continue;
        if (entry.is_directory())
            continue;
        filenames.push_back(filename);
    }

    return filenames;
}

InputFile create_input_file(std::string folder_path, std::string filename)
{
    std::ifstream input(folder_path + filename, std::ios::binary);

    if (input.fail())
        throw std::exception("The filename was invalid.");

    int width;
    int height;
    int byte_size;
    input.read(reinterpret_cast<char*>(&width), sizeof(width));
    input.read(reinterpret_cast<char*>(&height), sizeof(height));
    input.read(reinterpret_cast<char*>(&byte_size), sizeof(byte_size));
    if (byte_size != sizeof(short))
        throw std::exception("The depth pixels are not 16-bit.");

    return InputFile(filename, std::move(input), width, height);
}

// Converts 16-bit buffers into OpenCV Mats.
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
    cv::Mat chroma_channel(height, width, CV_8UC1, chroma_frame.data());

    std::vector<cv::Mat> y_cr_cb_channels;
    y_cr_cb_channels.push_back(y_channel);
    y_cr_cb_channels.push_back(chroma_channel);
    y_cr_cb_channels.push_back(chroma_channel);

    cv::Mat y_cr_cb_frame;
    cv::merge(y_cr_cb_channels, y_cr_cb_frame);

    cv::Mat bgr_frame = y_cr_cb_frame.clone();
    cvtColor(y_cr_cb_frame, bgr_frame, CV_YCrCb2BGR);
    return bgr_frame;
}

void run_trvl(InputFile& input_file, short change_threshold, int invalidation_threshold)
{
    int frame_size = input_file.width() * input_file.height();
    int depth_buffer_size = frame_size * sizeof(short);
    // For the raw pixels from the input file.
    std::vector<short> depth_buffer(frame_size);
    // For detecting changes and freezing other pixels.
    std::vector<trvl::Pixel> trvl_pixels(frame_size);
    // To save the pixel values of the previous frame to calculate differences between the previous and the current.
    std::vector<short> prev_pixel_values(frame_size);
    // The differences between the adjacent frames.
    std::vector<short> pixel_diffs(frame_size);
    // For the RVL compressed frame.
    std::vector<char> rvl_frame;
    // For the decompressed frame.
    std::vector<short> depth_image;

    int frame_count = 0;

    while (!input_file.input_stream().eof()) {
        input_file.input_stream().read(reinterpret_cast<char*>(depth_buffer.data()), depth_buffer_size);
        // Update the TRVL pixel values with the raw depth pixels.
        for (int i = 0; i < frame_size; ++i) {
            trvl::update_pixel(trvl_pixels[i], depth_buffer[i], change_threshold, invalidation_threshold);
        }

        // For the first frame, since there is no previous frame to diff, run vanilla RVL.
        if (frame_count == 0) {
            for (int i = 0; i < frame_size; ++i) {
                prev_pixel_values[i] = trvl_pixels[i].value();
            }
            rvl_frame = rvl::compress(prev_pixel_values.data(), frame_size);
            depth_image = rvl::decompress(rvl_frame.data(), frame_size);
        } else {
            // Calculate pixel_diffs using prev_pixel_values
            // and save current pixel values to prev_pixel_values for the next frame.
            for (int i = 0; i < frame_size; ++i) {
                short value = trvl_pixels[i].value();
                pixel_diffs[i] = value - prev_pixel_values[i];
                prev_pixel_values[i] = value;
            }
            // Compress and decompress the difference.
            rvl_frame = rvl::compress(pixel_diffs.data(), frame_size);
            auto diff_frame = rvl::decompress(rvl_frame.data(), frame_size);
            // Update depth_image of the previous frame using the difference
            // between the previous frame and the current frame.
            for (int i = 0; i < frame_size; ++i) {
                depth_image[i] += diff_frame[i];
            }
        }

        auto depth_mat = create_depth_mat(input_file.width(), input_file.height(), depth_image.data());

        cv::imshow("Depth", depth_mat);
        if (cv::waitKey(1) >= 0)
            return;

        ++frame_count;
    }
}

void run_video()
{
    short CHANGE_THRESHOLD = 10;
    int INVALIDATION_THRESHOLD = 2;
    const std::string DATA_FOLDER_PATH = "../../../data/";

    std::vector<std::string> filenames(get_filenames_from_folder_path(DATA_FOLDER_PATH));

    std::cout << "Input filenames inside the data folder:" << std::endl;
    for (int i = 0; i < filenames.size(); ++i) {
        std::cout << "\t(" << i << ") " << filenames[i] << std::endl;
    }

    int filename_index;
    for (;;) {
        std::cout << "Enter filename index: ";
        std::cin >> filename_index;
        if (filename_index >= 0 && filename_index < filenames.size())
            break;

        std::cout << "Invliad index." << std::endl;
    }

    std::string filename = filenames[filename_index];
    InputFile input_file(create_input_file(DATA_FOLDER_PATH, filename));

    run_trvl(input_file, CHANGE_THRESHOLD, INVALIDATION_THRESHOLD);
}

void main()
{
    for (;;)
        run_video();
}