#pragma once
#include <opencv2/core/core.hpp>
#include <omp.h>
#include <functional>
#include <algorithm>
#include <tuple>

#define data(im,x,y,c) im->imageData[i*im->widthStep+j*im->depth/8*im->nChannels+c]

inline void loopPixels(IplImage* im, std::function<void(unsigned char)> op, bool parallel = true)
{
	#pragma omp parallel for if (parallel)
	for (int i = 0; i < im->height; i++)
	{
		for (int j = 0; j < im->width; j++)
		{
			op(data(im, j, i, 0));
		}
	}
}

inline void loopPixelsApply(IplImage* im, std::function<unsigned char(unsigned char)> op, bool parallel = true)
{
	#pragma omp parallel for if (parallel)
	for (int i = 0; i < im->height; i++)
	{
		for (int j = 0; j < im->width; j++)
		{
			data(im, j, i, 0) = op(data(im, j, i, 0));
		}
	}
}

inline void loopPixelsRgb(IplImage* im, std::function<void(unsigned char, unsigned char, unsigned char)> op, bool parallel = true)
{
	#pragma omp parallel for if (parallel)
	for (int i = 0; i < im->height; i++)
	{
		for (int j = 0; j < im->width; j++)
		{
			op(data(im, j, i, 0), data(im, j, i, 1), data(im, j, i, 2));
		}
	}
}

inline void loopPixelsRgbApply(IplImage* im, std::function<std::tuple<unsigned char, unsigned char, unsigned char>(unsigned char, unsigned char, unsigned char)> op, bool parallel = true)
{
	#pragma omp parallel for if (parallel)
	for (int i = 0; i < im->height; i++)
	{
		for (int j = 0; j < im->width; j++)
		{
			auto rgb = op(data(im, j, i, 0), data(im, j, i, 1), data(im, j, i, 2));
			data(im, j, i, 0) = std::get<0>(rgb);
			data(im, j, i, 1) = std::get<1>(rgb);
			data(im, j, i, 2) = std::get<2>(rgb);
		}
	}
}

inline void showHistogram(IplImage* im, const char* title = "Histogram")
{
	using namespace cv;
	using namespace std;

	Mat b_hist, g_hist, r_hist, mat = cvarrToMat(im);

	vector<Mat> bgrs;
	split(mat, bgrs);

	int histSize = 256;

	float range[] = { 0, 256 };
	const float* histRange = { range };

	calcHist(&bgrs[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, true, false);
	calcHist(&bgrs[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, true, false);
	calcHist(&bgrs[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, true, false);

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound(double(hist_w) / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - b_hist.at<float>(i - 1)), Point(bin_w*(i), hist_h - b_hist.at<float>(i)), Scalar(255, 0, 0), 1, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - g_hist.at<float>(i - 1)), Point(bin_w*(i), hist_h - g_hist.at<float>(i)), Scalar(0, 255, 0), 1, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - r_hist.at<float>(i - 1)), Point(bin_w*(i), hist_h - r_hist.at<float>(i)), Scalar(0, 0, 255), 1, 8, 0);
	}

	namedWindow(title, CV_WINDOW_AUTOSIZE);
	imshow(title, histImage);
}
