#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "helpers.h"

using namespace cv;
using namespace std;

string read_file(string file)
{
	ifstream fs(file);
	string text((istreambuf_iterator<char>(fs)), istreambuf_iterator<char>());
	fs.close();
	return text;
}

void encode_lsb(IplImage* img, string text)
{
	int size = text.length();
	int xize = ~size;

	text = string(reinterpret_cast<char*>(&size), sizeof(int)) + string(reinterpret_cast<char*>(&xize), sizeof(int)) + text;

	int b = 0;
	int bits = text.length() * 8 + 7;

	for (int i = 0; i < img->height; i++)
	{
		for (int j = 0; j < img->width; j++)
		{
			for (int k = 0; k < img->nChannels; k++)
			{
				auto val = data(img, j, i, k);

				val &= 254;
				val |= (text[b / 8] & 1 << b % 8) >> b % 8;

				data(img, j, i, k) = val;

				if (b++ >= bits)
				{
					break;
				}
			}

			if (b >= bits)
			{
				break;
			}
		}

		if (b >= bits)
		{
			break;
		}
	}
}

string decode_lsb(IplImage* img)
{
	int b = 0;
	string text(img->height * img->width * img->nChannels / 8, 0);

	for (int i = 0; i < img->height; i++)
	{
		for (int j = 0; j < img->width; j++)
		{
			for (int k = 0; k < img->nChannels; k++)
			{
				auto val = data(img, j, i, k);

				text[b / 8] |= (val & 1) << b % 8;

				b++;
			}
		}
	}

	int size = *reinterpret_cast<const int*>(text.c_str());
	int xize = *reinterpret_cast<const int*>(text.c_str() + sizeof(int));

	if (xize != ~size)
	{
		return "";
	}

	return text.substr(sizeof(int) * 2, size);
}

void encode_lsb_alt(IplImage* img, string text)
{
	int size = text.length();
	int xize = ~size;

	text = string(reinterpret_cast<char*>(&size), sizeof(int)) + string(reinterpret_cast<char*>(&xize), sizeof(int)) + text;

	int b = 0;
	int bits = text.length() * 8;

	for (int i = 0; i < img->height; i += 3)
	{
		for (int j = 0; j < img->width; j += 3)
		{
			auto val = data(img, j, i, b % img->nChannels);

			val &= 254;
			val |= (text[b / 8] & 1 << b % 8) >> b % 8;

			data(img, j, i, b % img->nChannels) = val;

			if (b++ >= bits)
			{
				b = 0;
			}
		}

		if (b >= bits)
		{
			break;
		}
	}
}

string decode_lsb_alt(IplImage* img)
{
	int b = 0;
	string text((img->height / 3) * (img->width / 3) * (img->nChannels / 3) / 8, 0);

	for (int i = 0; i < img->height; i += 3)
	{
		for (int j = 0; j < img->width; j += 3)
		{
			auto val = data(img, j, i, b % img->nChannels);

			text[b / 8] |= (val & 1) << b % 8;

			b++;
		}
	}

	int size = *reinterpret_cast<const int*>(text.c_str());
	int xize = *reinterpret_cast<const int*>(text.c_str() + sizeof(int));

	if (xize != ~size)
	{
		return "";
	}

	return text.substr(sizeof(int) * 2, size);
}

int main(int argc, char** argv)
{
	auto img = cvLoadImage("img.png", CV_LOAD_IMAGE_COLOR);
	cvNamedWindow("Image", NULL);
	cvResizeWindow("Image", 1024, 768);
	//cvShowImage("Image", img);
	//showHistogram(img, "Pre-Steganography Histogram");

	encode_lsb_alt(img, read_file("test.txt"));
	cout << decode_lsb_alt(img);

	//showHistogram(img, "Post-Steganography Histogram");


	cvShowImage("Image", img);
	cvWaitKey();

	//system("pause");
	return EXIT_SUCCESS;
}