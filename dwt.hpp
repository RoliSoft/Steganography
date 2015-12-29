#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <tuple>

/*!
 * Performs Haar wavelet decomposition.
 *
 * \param src Source image.
 * \param dst Destination image for the decomposition.
 *
 * \return Tuple with the horizontal, vertical and diagonal coefficients.
 */
inline std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, std::vector<std::vector<float>>> cvHaarWavelet(const cv::Mat& src, cv::Mat& dst)
{
	using namespace cv;
	using namespace std;

	auto width  = src.cols / 2;
	auto height = src.rows / 2;

	vector<vector<float>> dhs(height);
	vector<vector<float>> dvs(height);
	vector<vector<float>> dds(height);

	for (int y = 0; y < height; y++)
	{
		dhs[y] = vector<float>(width);
		dvs[y] = vector<float>(width);
		dds[y] = vector<float>(width);

		for (int x = 0; x<(width);x++)
		{
			auto c  = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
			auto dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
			auto dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;
			auto dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1)) * 0.5;

			dst.at<float>(y,          x)         = c;
			dst.at<float>(y,          x + width) = dh;
			dst.at<float>(y + height, x)         = dv;
			dst.at<float>(y + height, x + width) = dd;

			dhs[y][x] = dh;
			dvs[y][x] = dv;
			dds[y][x] = dd;
		}
	}

	return make_tuple(dhs, dvs, dds);
}

/*!
 * Performs Haar wavelet reconstruction.
 *
 * \param src Source image.
 * \param dst Destination image for the reconstruction.
 * \param dhs Horizontal coefficients.
 * \param dvs Vertical coefficients.
 * \param dds Diagonal coefficients.
 */
inline void cvInvHaarWavelet(const cv::Mat& src, cv::Mat& dst, const std::vector<std::vector<float>>& dhs, const std::vector<std::vector<float>>& dvs, const std::vector<std::vector<float>>& dds)
{
	using namespace cv;
	using namespace std;

	auto width  = src.cols / 2;
	auto height = src.rows / 2;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			auto dh = dhs[y][x];
			auto dv = dvs[y][x];
			auto dd = dds[y][x];

			auto c = src.at<float>(y, x);

			dst.at<float>(y * 2,     x * 2)     = 0.5*(c + dh + dv + dd);
			dst.at<float>(y * 2,     x * 2 + 1) = 0.5*(c - dh + dv - dd);
			dst.at<float>(y * 2 + 1, x * 2)     = 0.5*(c + dh - dv - dd);
			dst.at<float>(y * 2 + 1, x * 2 + 1) = 0.5*(c - dh - dv + dd);
		}
	}
}

/*!
 * Uses discrete wavelet transformation to hide data in the diagonal filter of the first plane of an image.
 *
 * \param img Input image.
 * \param text Text to hide.
 * \param alpha Encoding intensity.
 *
 * \return Altered image with hidden data.
 */
inline cv::Mat encode_dwt(const cv::Mat& img, const std::string& text, float alpha = 0.05)
{
	using namespace cv;
	using namespace std;

	auto i = 0;
	auto size = text.length() * 8;

	Mat imgfp;
	img.convertTo(imgfp, CV_32F, 1.0 / 255);

	vector<Mat> planes;
	split(imgfp, planes);

	for (int y = 0;y < img.cols; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			auto val = planes[0].at<float>(y, x);

			if (val < alpha)
			{
				val = alpha;
			}
			else if (val > 1 - alpha)
			{
				val = 1 - alpha;
			}

			planes[0].at<float>(y, x) = val;
		}
	}

	Mat haar(img.rows, img.cols, CV_32FC1);

	auto hwv = cvHaarWavelet(planes[0], haar);
	auto dds = get<2>(hwv);

	for (int y = 0; y < dds.size(); y++)
	{
		for (int x = 0; x < dds[y].size(); x++)
		{
			auto val = 0;
			if (i <= size)
			{
				val = (text[i / 8] & 1 << i % 8) >> i % 8;
				i++;
			}

			if (val == 1)
			{
				dds[y][x] += alpha;
			}
			else
			{
				dds[y][x] -= alpha;
			}
		}
	}

	cvInvHaarWavelet(haar, planes[0], get<0>(hwv), get<1>(hwv), dds);

	Mat mergedfp;
	merge(planes, mergedfp);

	Mat merged;
	mergedfp.convertTo(merged, CV_8U, 255);

	return merged;
}

/*!
 * Uses discrete wavelet transformation to recover data hidden in the diagonal filter of an image.
 *
 * \param img Original image without hidden data.
 * \param stego Altered image with hidden data.
 *
 * \return Hidden data extracted form image.
 */
inline std::string decode_dwt(const cv::Mat& img, const cv::Mat& stego)
{
	using namespace cv;
	using namespace std;

	auto i = 0;
	string bits((img.cols / 2) * (img.rows  / 2) / 8, 0);

	Mat imgfp;
	img.convertTo(imgfp, CV_32FC1, 1.0 / 255);

	Mat stegofp;
	stego.convertTo(stegofp, CV_32FC1, 1.0 / 255);

	vector<Mat> planes1;
	split(imgfp, planes1);

	vector<Mat> planes2;
	split(stegofp, planes2);

	Mat haar1(img.rows, img.cols, CV_32FC1);
	Mat haar2(img.rows, img.cols, CV_32FC1);

	auto dds1 = get<2>(cvHaarWavelet(planes1[0], haar1));
	auto dds2 = get<2>(cvHaarWavelet(planes2[0], haar2));

	for (int y = 0; y < dds1.size(); y++)
	{
		for (int x = 0; x < dds1[y].size(); x++)
		{
			auto val = dds2[y][x] - dds1[y][x];

			if (val > 0)
			{
				bits[i / 8] |= 1 << i % 8;
			}

			i++;
		}
	}

	return bits;
}
