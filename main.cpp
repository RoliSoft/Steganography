#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "helpers.hpp"
#include "lsb.hpp"
#include "lsb_alt.hpp"
#include "dct.hpp"
#include "dwt.hpp"

using namespace cv;
using namespace std;

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

void test_lsb_alt()
{
	auto img = imread("img_small.png");

	namedWindow("Image", NULL);
	resizeWindow("Image", 512, 512);
	moveWindow("Image", 50, 50);
	imshow("Image", img);

	showHistogram(img, "Pre-Steganography Histogram");
	moveWindow("Pre-Steganography Histogram", 50, 595);

	auto input  = read_file("test.txt");
	auto stego  = encode_lsb_alt(img, input);
	auto output = decode_lsb_alt(stego);

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

void test_dwt()
{
	auto img = imread("lena.jpg");

	namedWindow("Image", NULL);
	resizeWindow("Image", 512, 512);
	moveWindow("Image", 50, 50);
	imshow("Image", img);

	showHistogram(img, "Pre-Steganography Histogram");
	moveWindow("Pre-Steganography Histogram", 50, 595);

	auto input  = read_file("test.txt");
	auto stego  = encode_dwt(img, input);
	auto output = decode_dwt(img, stego);

	cout << endl << "   Input:" << endl << endl << input << endl << endl << "   Extracted:" << endl << endl << output << endl;

	namedWindow("Altered Image", NULL);
	resizeWindow("Altered Image", 512, 512);
	moveWindow("Altered Image", 565, 50);
	imshow("Altered Image", stego);

	/*showHistogram(stego, "Post-Steganography Histogram");
	moveWindow("Post-Steganography Histogram", 565, 595);*/
}

int main(int argc, char** argv)
{
	//test_lsb();
	//test_lsb_alt();
	//test_dct();
	test_dwt();

	cvWaitKey();

	//system("pause");
	return EXIT_SUCCESS;
}