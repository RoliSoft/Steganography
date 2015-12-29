#pragma once
#include <opencv2/core/core.hpp>

/*!
 * Uses discrete cosine transformation to hide data in the coefficients of the first plane of an image.
 *
 * \param img Input image.
 * \param text Text to hide.
 * \param intensity Persistence of the hidden data.
 *
 * \return Altered image with hidden data.
 */
inline cv::Mat encode_dct(const cv::Mat& img, const std::string& text, int intensity = 25)
{
	using namespace cv;
	using namespace std;

	auto block_width  = 8;
	auto block_height = 8;
	auto grid_width   = img.cols / block_width;
	auto grid_height  = img.rows / block_height;

	auto i = 0;
	auto size = text.length() * 8;

	Mat imgfp;
	img.convertTo(imgfp, CV_64FC1);

	vector<Mat> planes;
	split(imgfp, planes);

	for (int x = 1; x < grid_width; x++)
	{
		for (int y = 1; y < grid_height; y++)
		{
			auto px = (x - 1) * block_width;
			auto py = (y - 1) * block_height;

			Mat block(planes[0], Rect(px, py, block_width, block_height));
			Mat trans(Size(block_width, block_height), block.type());

			dct(block, trans);

			auto a = trans.at<long float>(6, 7);
			auto b = trans.at<long float>(5, 1);
			
			auto val = 0;
			if (i <= size)
			{
				val = (text[i / 8] & 1 << i % 8) >> i % 8;
				i++;
			}

			if (val == 0)
			{
				if (a > b)
				{
					swap(a, b);
				}
			}
			else
			{
				if (a < b)
				{
					swap(a, b);
				}
			}

			if (a > b)
			{
				auto d = (intensity - (a - b)) / 2;
				     a = a + d;
				     b = b - d;
			}
			else
			{
				auto d = (intensity - (b - a)) / 2;
				     a = a - d;
				     b = b + d;
			}

			trans.at<long float>(6, 7) = a;
			trans.at<long float>(5, 1) = b;

			Mat stego(Size(block_width, block_height), block.type());

			idct(trans, stego);

			stego.copyTo(planes[0](Rect(px, py, block_width, block_height)));
		}
	}

	Mat mergedfp;
	merge(planes, mergedfp);

	Mat merged;
	mergedfp.convertTo(merged, CV_8U);

	return merged;
}

/*!
 * Uses discrete cosine transformation to recover data hidden in the coefficients of an image.
 *
 * \param img Input image with hidden data.
 *
 * \return Hidden data extracted form image.
 */
inline std::string decode_dct(const cv::Mat& img)
{
	using namespace cv;
	using namespace std;

	auto block_width  = 8;
	auto block_height = 8;
	auto grid_width   = img.cols / block_width;
	auto grid_height  = img.rows / block_height;

	auto i = 0;
	string bits(grid_width * grid_height / 8, 0);

	Mat imgfp;
	img.convertTo(imgfp, CV_64FC1);

	vector<Mat> planes;
	split(imgfp, planes);

	for (int x = 1; x < grid_width; x++)
	{
		for (int y = 1; y < grid_height; y++)
		{
			auto px = (x - 1) * block_width;
			auto py = (y - 1) * block_height;

			Mat block(planes[0], Rect(px, py, block_width, block_height));
			Mat trans(Size(block_width, block_height), block.type());

			dct(block, trans);

			auto a = trans.at<long float>(6, 7);
			auto b = trans.at<long float>(5, 1);

			if (a > b)
			{
				bits[i / 8] |= 1 << i % 8;
			}

			i++;
		}
	}

	return bits;
}
