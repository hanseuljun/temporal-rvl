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

void update_pixel(Pixel& pixel, short raw_value, int invalidation_threshold, short change_threshold);
}