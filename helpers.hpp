#pragma once
#include <opencv2/core/core.hpp>
#include <omp.h>
#include <fstream>
#include <functional>
#include <algorithm>

/*!
 * Reads the specified file into a string.
 *
 * \param file Path to the file.
 *
 * \return Contents of the file.
 */
inline std::string read_file(std::string file)
{
	std::ifstream fs(file);
	std::string text((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
	fs.close();
	return text;
}

/*!
 * Converts a cv::Mat object instance into an IplImage instance pointer.
 *
 * \param img Image in new format.
 *
 * \return Image in old format.
 */
inline IplImage* cvmatToArr(const cv::Mat& img)
{
	return new IplImage(img);
}

/*!
 * Displays the histogram of the specified image.
 *
 * \param mat Input image.
 * \param title Name of the window.
 */
inline void showHistogram(const cv::Mat& mat, const char* title = "Histogram")
{
	using namespace cv;
	using namespace std;

	Mat b_hist, g_hist, r_hist;

	vector<Mat> bgrs;
	split(mat, bgrs);

	auto size   = 256;
	auto hist_w = 512;
	auto hist_h = 400;
	auto bin_w  = cvRound(double(hist_w) / size);

	float range[] = { 0, 256 };
	const float* histRange = { range };

	calcHist(&bgrs[0], 1, 0, Mat(), b_hist, 1, &size, &histRange, true, false);
	calcHist(&bgrs[1], 1, 0, Mat(), g_hist, 1, &size, &histRange, true, false);
	calcHist(&bgrs[2], 1, 0, Mat(), r_hist, 1, &size, &histRange, true, false);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < size; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - b_hist.at<float>(i - 1)), Point(bin_w*(i), hist_h - b_hist.at<float>(i)), Scalar(255, 0, 0), 1, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - g_hist.at<float>(i - 1)), Point(bin_w*(i), hist_h - g_hist.at<float>(i)), Scalar(0, 255, 0), 1, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - r_hist.at<float>(i - 1)), Point(bin_w*(i), hist_h - r_hist.at<float>(i)), Scalar(0, 0, 255), 1, 8, 0);
	}

	namedWindow(title, CV_WINDOW_AUTOSIZE);
	imshow(title, histImage);
}
