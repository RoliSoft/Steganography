#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "helpers.hpp"
#include "lsb.hpp"
#include "dct.hpp"

using namespace cv;
using namespace std;

#define NONE 0  // no filter
#define HARD 1  // hard shrinkage
#define SOFT 2  // soft shrinkage
#define GARROT 3  // garrot filter

float sgn(float x)
{
	float res = 0;
	if (x == 0)
	{
		res = 0;
	}
	if (x>0)
	{
		res = 1;
	}
	if (x<0)
	{
		res = -1;
	}
	return res;
}

float soft_shrink(float d, float T)
{
	float res;
	if (fabs(d)>T)
	{
		res = sgn(d)*(fabs(d) - T);
	}
	else
	{
		res = 0;
	}

	return res;
}

float hard_shrink(float d, float T)
{
	float res;
	if (fabs(d)>T)
	{
		res = d;
	}
	else
	{
		res = 0;
	}

	return res;
}

float Garrot_shrink(float d, float T)
{
	float res;
	if (fabs(d)>T)
	{
		res = d - ((T*T) / d);
	}
	else
	{
		res = 0;
	}

	return res;
}

static void cvHaarWavelet(Mat &src, Mat &dst, int NIter)
{
	float c, dh, dv, dd;
	assert(src.type() == CV_32FC1);
	assert(dst.type() == CV_32FC1);
	int width = src.cols;
	int height = src.rows;
	for (int k = 0;k<NIter;k++)
	{
		for (int y = 0;y<(height >> (k + 1));y++)
		{
			for (int x = 0; x<(width >> (k + 1));x++)
			{
				c = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) + src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y, x) = c;

				dh = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y, x + (width >> (k + 1))) = dh;

				dv = (src.at<float>(2 * y, 2 * x) + src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) - src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y + (height >> (k + 1)), x) = dv;

				dd = (src.at<float>(2 * y, 2 * x) - src.at<float>(2 * y, 2 * x + 1) - src.at<float>(2 * y + 1, 2 * x) + src.at<float>(2 * y + 1, 2 * x + 1))*0.5;
				dst.at<float>(y + (height >> (k + 1)), x + (width >> (k + 1))) = dd;
			}
		}
		dst.copyTo(src);
	}
}

static void cvInvHaarWavelet(Mat &src, Mat &dst, int NIter, int SHRINKAGE_TYPE = 0, float SHRINKAGE_T = 50)
{
	float c, dh, dv, dd;
	assert(src.type() == CV_32FC1);
	assert(dst.type() == CV_32FC1);
	int width = src.cols;
	int height = src.rows;
	//--------------------------------
	// NIter - number of iterations 
	//--------------------------------
	for (int k = NIter;k>0;k--)
	{
		for (int y = 0;y<(height >> k);y++)
		{
			for (int x = 0; x<(width >> k);x++)
			{
				c = src.at<float>(y, x);
				dh = src.at<float>(y, x + (width >> k));
				dv = src.at<float>(y + (height >> k), x);
				dd = src.at<float>(y + (height >> k), x + (width >> k));

				// (shrinkage)
				switch (SHRINKAGE_TYPE)
				{
				case HARD:
					dh = hard_shrink(dh, SHRINKAGE_T);
					dv = hard_shrink(dv, SHRINKAGE_T);
					dd = hard_shrink(dd, SHRINKAGE_T);
					break;
				case SOFT:
					dh = soft_shrink(dh, SHRINKAGE_T);
					dv = soft_shrink(dv, SHRINKAGE_T);
					dd = soft_shrink(dd, SHRINKAGE_T);
					break;
				case GARROT:
					dh = Garrot_shrink(dh, SHRINKAGE_T);
					dv = Garrot_shrink(dv, SHRINKAGE_T);
					dd = Garrot_shrink(dd, SHRINKAGE_T);
					break;
				}

				//-------------------
				dst.at<float>(y * 2, x * 2) = 0.5*(c + dh + dv + dd);
				dst.at<float>(y * 2, x * 2 + 1) = 0.5*(c - dh + dv - dd);
				dst.at<float>(y * 2 + 1, x * 2) = 0.5*(c + dh - dv - dd);
				dst.at<float>(y * 2 + 1, x * 2 + 1) = 0.5*(c - dh - dv + dd);
			}
		}
		Mat C = src(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
		Mat D = dst(Rect(0, 0, width >> (k - 1), height >> (k - 1)));
		D.copyTo(C);
	}
}

void encode_dwt(IplImage* img, string secret_bin, float alpha = 0.05)
{
	int secret_msg_bin_len = secret_bin.length();

	auto car = cvarrToMat(img);
	Mat carf;
	car.convertTo(carf, CV_32FC1, 1.0 / 255); // 1.0/255

	/*double M = 0, m = 0;
	minMaxLoc(carf, &m, &M);
	if ((M - m) > 0)
	{
		carf = carf * (1.0 / (M - m)) - m / (M - m);
	}*/

	Mat GrayFrame = Mat(carf.rows, carf.cols, CV_8UC1);
	Mat Src = Mat(carf.rows, carf.cols, CV_32FC1);
	Mat Dst = Mat(carf.rows, carf.cols, CV_32FC1);
	Mat Temp = Mat(carf.rows, carf.cols, CV_32FC1);
	Mat Filtered = Mat(carf.rows, carf.cols, CV_32FC1);

	cvtColor(carf, GrayFrame, CV_BGR2GRAY);
	GrayFrame.convertTo(Src, CV_32FC1);
	cvHaarWavelet(Src, Dst, 1);
	Dst.copyTo(Temp);
	cvInvHaarWavelet(Temp, Filtered, 1, HARD, 30);

	cvShowImage("Image2", cvmatToArr(Filtered));

	/*vector<Mat> planes;
	split(carf, planes);*/
}

void test_lsb()
{
	auto img = imread("img_small.png");

	namedWindow("Image", NULL);
	resizeWindow("Image", 512, 512);
	moveWindow("Image", 50, 50);
	imshow("Image", img);

	showHistogram(img, "Pre-Steganography Histogram");
	moveWindow("Pre-Steganography Histogram", 50, 595);

	auto input  = read_file("test.txt");
	auto stego  = encode_lsb(img, input);
	auto output = decode_lsb(stego);

	cout << endl << "   Input:" << endl << endl << input << endl << endl << "   Extracted:" << endl << endl << output << endl;

	namedWindow("Altered Image", NULL);
	resizeWindow("Altered Image", 512, 512);
	moveWindow("Altered Image", 565, 50);
	imshow("Altered Image", stego);

	showHistogram(stego, "Post-Steganography Histogram");
	moveWindow("Post-Steganography Histogram", 565, 595);
}

void test_dct()
{
	auto img = imread("lena.jpg");

	namedWindow("Image", NULL);
	resizeWindow("Image", 512, 512);
	moveWindow("Image", 50, 50);
	imshow("Image", img);

	showHistogram(img, "Pre-Steganography Histogram");
	moveWindow("Pre-Steganography Histogram", 50, 595);

	auto input  = read_file("test.txt");
	auto stego  = encode_dct(img, input);
	auto output = decode_dct(stego);

	cout << endl << "   Input:" << endl << endl << input << endl << endl << "   Extracted:" << endl << endl << output << endl;

	namedWindow("Altered Image", NULL);
	resizeWindow("Altered Image", 512, 512);
	moveWindow("Altered Image", 565, 50);
	imshow("Altered Image", stego);

	showHistogram(stego, "Post-Steganography Histogram");
	moveWindow("Post-Steganography Histogram", 565, 595);
}

int main(int argc, char** argv)
{
	test_lsb();
	//test_dct();

	cvWaitKey();

	//system("pause");
	return EXIT_SUCCESS;
}