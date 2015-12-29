#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "format.h"
#include "helpers.hpp"
#include "lsb.hpp"
#include "lsb_alt.hpp"
#include "dct.hpp"
#include "dwt.hpp"
#include "tlv.hpp"

using namespace cv;
using namespace std;
using namespace boost;

/*!
 * Evaluates the similarity and prints the original and resulting strings.
 *
 * \param input Original input.
 * \param output Extracted output.
 */
void print_debug(const string& input, const string& output)
{
	auto original  = clean(input);
	auto extracted = clean(output);
	auto accuracy  = similarity(original, extracted);

	Format::ColorCode color;

	if (accuracy > 99.99)
	{
		color = Format::Green;
	}
	else if (accuracy > 75)
	{
		color = Format::Yellow;
	}
	else
	{
		color = Format::Red;
	}

	cout << endl
		 << "   Similarity: " << setprecision(3) << color << Format::Bold << accuracy << "%" << Format::Normal << Format::Default << endl << endl
		 << "   Input:"       << endl << endl << Format::White << Format::Bold << original  << Format::Normal << Format::Default << endl << endl
		 << "   Extracted:"   << endl << endl << Format::White << Format::Bold << extracted << Format::Normal << Format::Default << endl;
}

/*!
 * Displays the original image and pre-steganography histogram.
 */
void show_image(const Mat& img, const string& modifier = "", bool histogram = true)
{
	static auto i = 0;

	i++;

	auto title = modifier.empty() ? "Image " + to_string(i) : modifier + " Image";

	namedWindow(title, NULL);
	resizeWindow(title, 512, 512);
	moveWindow(title, 50 + (515 * (i - 1)), 50);
	imshow(title, img);

	if (histogram)
	{
		title = modifier.empty() ? "Histogram " + to_string(i) : modifier + " Histogram";

		showHistogram(img, title.c_str());
		moveWindow(title, 50 + (515 * (i - 1)), 595);
	}
}

/*!
 * Tests the least significat bit method.
 */
void test_lsb()
{
	auto img = imread("img_small.png");

	show_image(img, "Original");

	auto input  = read_file("test.txt");
	auto stego  = encode_lsb(img, encode_tlv(input));
	auto output = decode_tlv(decode_lsb(stego));

	print_debug(input, output);

	show_image(stego, "Altered");
}

/*!
 * Tests the alternate least significat bit method.
 */
void test_lsb_alt()
{
	auto img = imread("img_small.png");

	show_image(img, "Original");

	auto input  = read_file("test.txt");
	auto stego  = encode_lsb_alt(img, encode_tlv(input));
	auto output = decode_tlv(decode_lsb_alt(stego));

	print_debug(input, output);

	show_image(stego, "Altered");
}

/*!
 * Tests the discrete cosine transformation method.
 */
void test_dct()
{
	auto img = imread("lena.jpg");

	show_image(img, "Original");

	auto input  = read_file("test.txt");
	auto stego  = encode_dct(img, input);
	auto output = decode_dct(stego);

	print_debug(input, output);

	show_image(stego, "Altered");
}

/*!
 * Tests the discrete cosine transformation method with 80% JPEG compression
 * and multi-channel message reconstruction.
 */
void test_dct_multi()
{
	auto img = imread("lena.jpg");

	show_image(img, "Original");

	auto input = read_file("test.txt");
	auto stego = encode_dct(img,   input, STORE_FULL, 0);
	     stego = encode_dct(stego, input, STORE_FULL, 1);
		 stego = encode_dct(stego, input, STORE_FULL, 2);

	imwrite("lena_dct.jpg", stego, vector<int> { CV_IMWRITE_JPEG_QUALITY, 80 });
	stego = imread("lena_dct.jpg");

	auto output = repair(vector<string>
		{
			decode_dct(stego, 0),
			decode_dct(stego, 1),
			decode_dct(stego, 2)
		});

	print_debug(input, output);

	show_image(stego, "Altered");
}

/*!
 * Tests the discrete wavelet transformation method.
 */
void test_dwt()
{
	auto img = imread("lena.jpg");

	show_image(img, "Original");

	auto input  = read_file("test.txt");
	auto stego  = encode_dwt(img, input);
	auto output = decode_dwt(img, stego);

	print_debug(input, output);

	show_image(stego, "Altered");
}

/*!
 * Tests the discrete wavelet transformation method with 90% JPEG compression
 * and multi-channel message reconstruction.
 */
void test_dwt_multi()
{
	auto img = imread("lena.jpg");

	show_image(img, "Original");

	auto input = read_file("test.txt");
	auto stego = encode_dwt(img,   input, STORE_FULL, 0);
	     stego = encode_dwt(stego, input, STORE_FULL, 1);
		 stego = encode_dwt(stego, input, STORE_FULL, 2);

	imwrite("lena_dwt.jpg", stego, vector<int> { CV_IMWRITE_JPEG_QUALITY, 90 });
	stego = imread("lena_dwt.jpg");

	auto output = repair(vector<string>
		{
			decode_dwt(img, stego, 0),
			decode_dwt(img, stego, 1),
			decode_dwt(img, stego, 2)
		});

	print_debug(input, output);

	show_image(stego, "Altered");
}

/*!
 * Entry point of the application.
 *
 * \param Number of arguments.
 * \param Argument array pointer.
 *
 * \return Value indicating exit status.
 */
int main(int argc, char** argv)
{
	Format::Init();

	//test_lsb();
	//test_lsb_alt();
	//test_dct();
	test_dct_multi();
	//test_dwt();
	//test_dwt_multi();

	cvWaitKey();

	//system("pause");
	return EXIT_SUCCESS;
}
