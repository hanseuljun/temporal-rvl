#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "trvl.h"

class InputFile
{
public:
	InputFile(std::ifstream&& input_stream, int width, int height)
		: input_stream_(std::move(input_stream)), width_(width), height_(height) {}

	std::ifstream& input_stream() { return input_stream_; }
	int width() { return width_; }
	int height() { return height_; }

private:
	std::ifstream input_stream_;
	int width_;
	int height_;
};

class Timer
{
public:
	Timer() : time_point_(std::chrono::high_resolution_clock::now()) {}
	float milliseconds()
	{
		auto now = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::microseconds>(now - time_point_).count() / 1000.0f;
	}

private:
	std::chrono::steady_clock::time_point time_point_;
};

class Result
{
public:
	Result(float average_time, float average_frame_size_ratio, float average_psnr)
		: average_time_(average_time), average_frame_size_ratio_(average_frame_size_ratio),
		  average_psnr_(average_psnr) {}

	float average_time() { return average_time_; }
	float average_frame_size_ratio() { return average_frame_size_ratio_; }
	float average_psnr() { return average_psnr_; }

private:
	float average_time_;
	float average_frame_size_ratio_;
	float average_psnr_;
};

std::vector<std::string> get_filenames_from_folder_path(std::string folder_path) {
	std::vector<std::string> filenames;
	for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
		std::string filename = std::filesystem::path(entry.path()).filename().string();
		if (filename == ".gitignore")
			continue;
		filenames.push_back(filename);
	}

	return filenames;
}

InputFile create_input_file(std::string file_path) {
	std::ifstream input(file_path, std::ios::binary);

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

	return InputFile(std::move(input), width, height);
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

void write_result_output_line(std::ofstream& result_output, std::string filename, std::string type, Result result) {
	result_output << filename << ", "
				  << type << ", "
				  << result.average_time() << ", "
				  << result.average_frame_size_ratio() << ", "
				  << result.average_psnr() << std::endl;
}

Result run_rvl(InputFile& input_file)
{
    int frame_size = input_file.width() * input_file.height();
    int depth_buffer_size = frame_size * sizeof(short);
	// For the raw pixels from the input file.
    std::vector<short> depth_buffer(frame_size);
	// For the RVL compressed frame.
	std::vector<char> rvl_frame;
	// For the decompressed frame.
	std::vector<short> depth_image;

    float time_sum = 0.0f;
    int compressed_size_sum = 0;
    int frame_count = 0;
    while (!input_file.input_stream().eof()) {
		input_file.input_stream().read(reinterpret_cast<char*>(depth_buffer.data()), depth_buffer_size);

        Timer timer;
        rvl_frame = rvl::compress(depth_buffer.data(), frame_size);
        depth_image = rvl::decompress(rvl_frame.data(), frame_size);
        time_sum += timer.milliseconds();

        auto depth_mat = create_depth_mat(input_file.width(), input_file.height(), depth_image.data());

        cv::imshow("Depth", depth_mat);
        if (cv::waitKey(1) >= 0)
            break;

        compressed_size_sum += rvl_frame.size();
        ++frame_count;
    }

    float average_time = time_sum / frame_count;
    float average_frame_size_ratio = compressed_size_sum / (float) (depth_buffer_size * frame_count);
    std::cout << "RVL" << std::endl
              << "average time: " << average_time << " ms" << std::endl
              << "average frame size ratio: " << average_frame_size_ratio << std::endl;

	return Result(average_time, average_frame_size_ratio, 0.0f);
}

Result run_trvl(InputFile& input_file, int invalidation_threshold, short change_threshold)
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

    float time_sum = 0.0f;
    int compressed_size_sum = 0;
    float psnr_sum = 0.0f;
	int zero_psnr_frame_count = 0;
    int frame_count = 0;

    while (!input_file.input_stream().eof()) {
		input_file.input_stream().read(reinterpret_cast<char*>(depth_buffer.data()), depth_buffer_size);
        Timer timer;
		// Update the TRVL pixel values with the raw depth pixels.
        for (int i = 0; i < frame_size; ++i) {
			trvl::update_pixel(trvl_pixels[i], depth_buffer[i], invalidation_threshold, change_threshold);
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
		time_sum += timer.milliseconds();

        auto depth_mat = create_depth_mat(input_file.width(), input_file.height(), depth_image.data());

        cv::imshow("Depth", depth_mat);
        if (cv::waitKey(1) >= 0)
            break;

        compressed_size_sum += rvl_frame.size();
		// The first frame goes through vanilla RVL which is lossless.
		float mse_value = mse(depth_buffer, depth_image);
		if (mse_value != 0.0f) {
			psnr_sum += max(depth_buffer) / sqrt(mse_value);
		} else {
			++zero_psnr_frame_count;
		}
		++frame_count;
    }

    float average_time = time_sum / frame_count;
    float average_frame_size_ratio = compressed_size_sum / (float) (depth_buffer_size * frame_count);
    float average_psnr = psnr_sum / (frame_count - zero_psnr_frame_count);
    std::cout << "Temporal RVL" << std::endl
              << "average time: " << average_time << " ms" << std::endl
              << "average frame size ratio: " << average_frame_size_ratio << std::endl
              << "average PSNR: " << average_psnr << std::endl;

	return Result(average_time, average_frame_size_ratio, average_psnr);
}

void run_one_video()
{
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
    std::string file_path = DATA_FOLDER_PATH + filename;
	InputFile input_file(create_input_file(file_path));

	for (;;) {
		std::cout << "Enter compression type (0: RVL, 1: TRVL): ";
		std::string compression_type;
		std::cin >> compression_type;

		if (compression_type == "0") {
			run_rvl(input_file);
			break;
		} else if (compression_type == "1") {
			int INVALIDATION_THRESHOLD = 2;
			short CHANGE_THRESHOLD = 10;
			run_trvl(input_file, INVALIDATION_THRESHOLD, CHANGE_THRESHOLD);
			break;
		} else {
			std::cout << "Invalid compression type." << std::endl;
		}
	}
}

void run_all_videos()
{
	const std::string DATA_FOLDER_PATH = "../../../data/";
	const std::string RESULT_OUTPUT_FILE_PATH = "../../../output/result.csv";
	std::vector<std::string> filenames(get_filenames_from_folder_path(DATA_FOLDER_PATH));
	std::ofstream result_output(RESULT_OUTPUT_FILE_PATH, std::ios::out);
	result_output << "filename, type, time, frame_size_ratio, psnr" << std::endl;

	for (auto& filename : filenames) {
		std::string file_path = DATA_FOLDER_PATH + filename;
		Result rvl_result(run_rvl(create_input_file(file_path)));

		int INVALIDATION_THRESHOLD = 2;
		short CHANGE_THRESHOLD = 10;
		Result trvl_result(run_trvl(create_input_file(file_path), INVALIDATION_THRESHOLD, CHANGE_THRESHOLD));

		write_result_output_line(result_output, filename, "rvl", rvl_result);
		write_result_output_line(result_output, filename, "trvl", trvl_result);
	}
}

void main()
{
	std::cout << "Run all videos? (y/n) ";
	std::string input;
	std::cin >> input;

	if (input == "y") {
		run_all_videos();
		return;
	}

	for (;;)
		run_one_video();
}