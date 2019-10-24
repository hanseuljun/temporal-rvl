#pragma once

#include <vector>

// This algorithm is from
// Wilson, A. D. (2017, October). Fast lossless depth image compression.
// In Proceedings of the 2017 ACM International Conference on Interactive Surfaces and Spaces (pp. 100-105). ACM.
namespace rvl
{
std::vector<char> compress(short* input, int num_pixels);
std::vector<short> decompress(char* input, int num_pixels);
}