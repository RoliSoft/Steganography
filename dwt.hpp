#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <tuple>

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
			auto c  = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
			auto dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
			auto dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
			auto dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;

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

	Mat plane = planes[0];

	for (int y = 0;y < plane.cols; y++)
	{
		for (int x = 0; x < plane.cols; x++)
		{
			auto val = plane.at<float>(y, x);

			if (val < alpha)
			{
				val = alpha;
			}
			else if (val > 1 - alpha)
			{
				val = 1 - alpha;
			}

			plane.at<float>(y, x) = val;
		}
	}

	Mat Dst = Mat(plane.rows, plane.cols, CV_32FC1);
	Mat Temp = Mat(plane.rows, plane.cols, CV_32FC1);
	Mat Filtered = Mat(plane.rows, plane.cols, CV_32FC1);

	auto params = cvHaarWavelet(plane, Dst);
	auto dds = get<2>(params);

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

	Dst.copyTo(Temp);

	cvInvHaarWavelet(Temp, Filtered, get<0>(params), get<1>(params), dds);

	/*imshow("Image Haar", Dst);
	imshow("Image Inv", Filtered);*/

	Filtered.copyTo(planes[0]);

	Mat mergedfp;
	merge(planes, mergedfp);

	/*Mat merged;
	mergedfp.convertTo(merged, CV_8U);*/

	//imshow("Image Res", mergedfp);
	return mergedfp;
}

std::string decode_dwt(const cv::Mat& img, const cv::Mat& stego)
{
	using namespace cv;
	using namespace std;

	auto i = 0;
	string bits((img.cols / 2) * (img.rows  / 2) / 8, 0);

	Mat imgfp;
	img.convertTo(imgfp, CV_32FC1, 1.0 / 255);

	Mat stegofp;
	stego.convertTo(stegofp, CV_32FC1);

	vector<Mat> planes1;
	split(imgfp, planes1);

	vector<Mat> planes2;
	split(stegofp, planes2);

	Mat plane1 = planes1[0];
	Mat plane2 = planes2[0];

	Mat Dst1 = Mat(img.rows, img.cols, CV_32FC1);
	Mat Dst2 = Mat(img.rows, img.cols, CV_32FC1);

	auto ddo = get<2>(cvHaarWavelet(plane1, Dst1));
	auto dds = get<2>(cvHaarWavelet(plane2, Dst2));

	for (int y = 0; y < ddo.size(); y++)
	{
		for (int x = 0; x < ddo[y].size(); x++)
		{
			auto val = dds[y][x] - ddo[y][x];

			if (val > 0)
			{
				bits[i / 8] |= 1 << i % 8;
			}

			i++;
		}
	}

	return bits;
}
