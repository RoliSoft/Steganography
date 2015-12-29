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

using namespace cv;
using namespace std;
using namespace boost;

/*!
 * Returns the similarity between the original message and extracted message.
 *
 * \param original Original data.
 * \param extracted Extracted data.
 *
 * \return Similarity in percentages.
 */
float similarity(const string& original, const string& extracted)
{
	auto hits = 0;

	for (int i = 0; i < min(original.length(), extracted.length()); i++)
	{
		if (original[i] == extracted[i])
		{
			hits++;
		}
	}

	return float(hits) / original.length() * 100;
}

void repair()
{
	
}

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

	auto color = Format::White;

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
	auto stego  = encode_lsb(img, input);
	auto output = decode_lsb(stego);

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
	auto stego  = encode_lsb_alt(img, input);
	auto output = decode_lsb_alt(stego);

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
	test_dct();
	//test_dwt();

	cvWaitKey();

	//system("pause");
	return EXIT_SUCCESS;
}
