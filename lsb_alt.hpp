#pragma once
#include <opencv2/core/core.hpp>

/*!
 * Hides data in an image by manipulating the least significant bits of each pixel.
 * This version does not utilize all the channels and sequentially hops between them.
 *
 * \param img Input image.
 * \param text Text to hide.
 *
 * \return Altered image with hidden data.
 */
inline cv::Mat encode_lsb_alt(const cv::Mat& img, const std::string& text)
{
	using namespace cv;
	using namespace std;

	int size = text.length();
	int xize = ~size;

	auto data = string(reinterpret_cast<char*>(&size), sizeof(int)) + string(reinterpret_cast<char*>(&xize), sizeof(int)) + text;

	int b = 0;
	int bits = data.length() * 8 + 7;

	Mat stego;
	img.copyTo(stego);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			auto val = img.at<Vec3b>(i, j)[b % img.channels()];

			val &= 254;
			val |= (data[b / 8] & 1 << b % 8) >> b % 8;

			stego.at<Vec3b>(i, j)[b % img.channels()] = val;

			if (b++ >= bits)
			{
				break;
			}
		}

		if (b >= bits)
		{
			break;
		}
	}

	return stego;
}

/*!
 * Recovers data hidden in an image using least significant bit manipulation.
 * This version does not utilize all the channels and sequentially hops between them.
 *
 * \param img Input image with hidden data.
 *
 * \return Hidden data extracted form image.
 */
inline std::string decode_lsb_alt(const cv::Mat& img)
{
	using namespace cv;
	using namespace std;

	auto b = 0;
	string text(img.cols * img.rows * img.channels() / 8, 0);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			auto val = img.at<Vec3b>(i, j)[b % img.channels()];

			text[b / 8] |= (val & 1) << b % 8;

			b++;
		}
	}

	auto size = *reinterpret_cast<const int*>(text.c_str());
	auto xize = *reinterpret_cast<const int*>(text.c_str() + sizeof(int));

	if (xize != ~size)
	{
		return "";
	}

	return text.substr(sizeof(int) * 2, size);
}
