#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <tuple>

inline std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, std::vector<std::vector<float>>> cvHaarWavelet(cv::Mat &src, cv::Mat &dst)
{
	using namespace cv;
	using namespace std;

	float c, dh, dv, dd;
	assert(src.type() == CV_32FC1);
	assert(dst.type() == CV_32FC1);
	int width = src.cols;
	int height = src.rows;

	vector<vector<float>> dhs(height >> 1);
	vector<vector<float>> dvs(height >> 1);
	vector<vector<float>> dds(height >> 1);

	for (int y = 0;y<(height >> 1);y++)
	{
		dhs[y] = vector<float>(width >> 1);
		dvs[y] = vector<float>(width >> 1);
		dds[y] = vector<float>(width >> 1);

		for (int x = 0; x<(width >> 1);x++)
		{
			c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
			dst.at<float>(y, x) = c;

			dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
			dst.at<float>(y, x + (width >> 1)) = dh;

			dhs[y][x] = dh;

			dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
			dst.at<float>(y + (height >> 1), x) = dv;

			dvs[y][x] = dv;

			dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
			dst.at<float>(y + (height >> 1), x + (width >> 1)) = dd;

			dds[y][x] = dd;
		}
	}
	dst.copyTo(src);

	return make_tuple(dhs, dvs, dds);
}

inline void cvInvHaarWavelet(cv::Mat &src, cv::Mat &dst, std::vector<std::vector<float>>& dhs, std::vector<std::vector<float>>& dvs, std::vector<std::vector<float>>& dds)
{
	using namespace cv;
	using namespace std;

	float c, dh, dv, dd;
	assert(src.type() == CV_32FC1);
	assert(dst.type() == CV_32FC1);
	int width = src.cols;
	int height = src.rows;

	for (int y = 0;y<(height >> 1);y++)
	{
		for (int x = 0; x<(width >> 1);x++)
		{
			c = src.at<float>(y, x);

			dh = dhs[y][x];
			dv = dvs[y][x];
			dd = dds[y][x];

			dst.at<float>(y * 2, x * 2) = 0.5*(c + dh + dv + dd);
			dst.at<float>(y * 2, x * 2 + 1) = 0.5*(c - dh + dv - dd);
			dst.at<float>(y * 2 + 1, x * 2) = 0.5*(c + dh - dv - dd);
			dst.at<float>(y * 2 + 1, x * 2 + 1) = 0.5*(c - dh - dv + dd);
		}
	}

	Mat C = src(Rect(0, 0, width >> (1 - 1), height >> (1 - 1)));
	Mat D = dst(Rect(0, 0, width >> (1 - 1), height >> (1 - 1)));
	D.copyTo(C);
}

inline cv::Mat encode_dwt(const cv::Mat& img, const std::string& text, float alpha = 0.05)
{
	using namespace cv;
	using namespace std;

	auto i = 0;
	auto size = text.length() * 8;

	Mat imgfp;
	img.convertTo(imgfp, CV_32FC1, 1.0 / 255); // 1.0/255

	for (int y = 0;y < imgfp.cols;y++)
	{
		for (int x = 0; x < imgfp.cols;x++)
		{
			auto val = imgfp.at<float>(y, x);

			if (val < alpha)
			{
				val = alpha;
			}
			else if (val > 1 - alpha)
			{
				val = 1 - alpha;
			}

			imgfp.at<float>(y, x) = val;
		}
	}

	Mat GrayFrame = Mat(imgfp.rows, imgfp.cols, CV_8UC1);
	Mat Src = Mat(imgfp.rows, imgfp.cols, CV_32FC1);
	Mat Dst = Mat(imgfp.rows, imgfp.cols, CV_32FC1);
	Mat Temp = Mat(imgfp.rows, imgfp.cols, CV_32FC1);
	Mat Filtered = Mat(imgfp.rows, imgfp.cols, CV_32FC1);

	cvtColor(imgfp, GrayFrame, CV_BGR2GRAY);
	GrayFrame.convertTo(Src, CV_32FC1);

	auto params = cvHaarWavelet(Src, Dst);
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

	//imshow("Image Haar", Dst);
	imshow("Image Inv", Filtered);

	/*vector<Mat> planes;
	split(carf, planes);*/

	return Filtered;
}

std::string decode_dwt(const cv::Mat& img, const cv::Mat& stego)
{
	using namespace cv;
	using namespace std;

	auto i = 0;
	string bits((img.cols >> 1) * (img.rows >> 1) / 8, 0);

	Mat imgfp;
	img.convertTo(imgfp, CV_32FC1, 1.0 / 255);

	/*Mat stegofp;
	stego.convertTo(stegofp, CV_32FC1, 1.0 / 255);*/

	Mat GrayFrame1 = Mat(imgfp.rows, imgfp.cols, CV_8UC1);
	//Mat GrayFrame2 = Mat(imgfp.rows, imgfp.cols, CV_8UC1);
	Mat Src1 = Mat(imgfp.rows, imgfp.cols, CV_32FC1);
	Mat Dst1 = Mat(imgfp.rows, imgfp.cols, CV_32FC1);
	Mat Src2 = Mat(imgfp.rows, imgfp.cols, CV_32FC1);
	Mat Dst2 = Mat(imgfp.rows, imgfp.cols, CV_32FC1);
	cvtColor(imgfp, GrayFrame1, CV_BGR2GRAY);
	GrayFrame1.convertTo(Src1, CV_32FC1);
	/*cvtColor(stegofp, GrayFrame2, CV_BGR2GRAY);
	GrayFrame2.convertTo(Src2, CV_32FC1);*/
	Src2 = stego;

	auto ddo = get<2>(cvHaarWavelet(Src1, Dst1));
	auto dds = get<2>(cvHaarWavelet(Src2, Dst2));

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
