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

IplImage* encode_dct(IplImage* carrier, string secret_bin, int persistence = 100)
{
	int secret_bin_i = 0;
	int secret_length = secret_bin.length() * 8;

	int block_width = 8;
	int block_height = 8;

	// 3 6
	// 5 2

	int s1x = 3;
	int s1y = 6;
	int s2x = 5;
	int s2y = 2;

	int width = carrier->width;
	int height = carrier->height;

	int grid_width = width / block_width;
	int grid_height = height / block_height;

	auto car = cvarrToMat(carrier);
	Mat carf;
	car.convertTo(carf, CV_64FC1); // 1.0/255
	vector<Mat> planes;
	split(carf, planes);

	for (int gx = 1; gx < (grid_width / 2); gx++)
	{
		for (int gy = 1; gy < (grid_height / 2); gy++)
		{
			int cx = (gx - 1) * block_width;
			int cy = (gy - 1) * block_height;

			int posx = cx + block_width;
			int posy = cy + block_height;

			Mat block(planes[0], Rect(cx, cy, posx, posy));
			Mat blout(Size(block_width, block_height), block.type());
			dct(block, blout);
			//auto block = dct2(carrier[posx, posy]);

			auto c1 = blout.at<long float>(s1x, s1y);
			auto c2 = blout.at<long float>(s2x, s2y);
			
			int secret_bit;
			if (secret_bin_i <= secret_length)
			{
				secret_bit = (secret_bin[secret_bin_i / 8] & 1 << secret_bin_i % 8) >> secret_bin_i % 8;
				//secret_bit = (secret_bin[floor(secret_bin_i / 8)] & (1 << secret_bin_i % 8)) == (1 << secret_bin_i % 8) ? 1 : 0;
			}
			else
			{
				secret_bit = 0;
			}

			secret_bin_i++;
			cout << secret_bit;
			if (secret_bit == 0)
			{
				if (c1 > c2)
				{
					swap(c1, c2);
				}
			}
			else
			{
				if (c1 < c2)
				{
					swap(c1, c2);
				}
			}

			if (c1 > c2)
			{
				auto d = (persistence - (c1 - c2)) / 2;
				    c1 = c1 + d;
				    c2 = c2 - d;
			}
			else
			{
				auto d = (persistence - (c2 - c1)) / 2;
				    c1 = c1 - d;
				    c2 = c2 + d;
			}

			blout.at<long float>(s1x, s1y) = c1;
			blout.at<long float>(s2x, s2y) = c2;

			Mat blsteg(Size(block_width, block_height), block.type());
			idct(blout, blsteg);

			blsteg.copyTo(planes[0](Rect(cx, cy, posx, posy)));
		}
	}

	Mat merged;
	merge(planes, merged);

	Mat mergedi;
	merged.convertTo(mergedi, CV_8U);

	return cvCloneImage(&(IplImage)mergedi);
}

string decode_dct(IplImage* stego)
{
	int block_width = 8;
	int block_height = 8;

	// 3 6
	// 5 2

	int s1x = 3;
	int s1y = 6;
	int s2x = 5;
	int s2y = 2;

	int width = stego->width;
	int height = stego->height;

	int grid_width = width / block_width;
	int grid_height = height / block_height;

	//char* stego_bin = new char[(grid_width * grid_height) / 8];
	string stego_bin((grid_width * grid_height) / 8, 0);
	int stego_bin_i = 0;

	auto car = cvarrToMat(stego);
	Mat carf;
	car.convertTo(carf, CV_64FC1); // 1.0/255
	vector<Mat> planes;
	split(carf, planes);

	for (int gx = 1; gx < (grid_width / 2); gx++)
	{
		for (int gy = 1; gy < (grid_height / 2); gy++)
		{
			int cx = (gx - 1) * block_width;
			int cy = (gy - 1) * block_height;

			int posx = cx + block_width;
			int posy = cy + block_height;

			Mat block(planes[0], Rect(cx, cy, posx, posy));
			Mat blout(Size(block_width, block_height), block.type());
			dct(block, blout);

			auto c1 = blout.at<long float>(s1x, s1y);
			auto c2 = blout.at<long float>(s2x, s2y);

			if (c1 > c2)
			{
				stego_bin[stego_bin_i / 8] |= 1 << stego_bin_i % 8;
			}
			else
			{
				//stego_bin[stego_bin_i] = 0;
			}

			stego_bin_i++;
		}
	}

	return stego_bin;
}

int main(int argc, char** argv)
{
	auto img = cvLoadImage("img_small.png", CV_LOAD_IMAGE_COLOR);
	cvNamedWindow("Image", NULL);
	cvResizeWindow("Image", 640, 400);
	//cvShowImage("Image", img);
	//showHistogram(img, "Pre-Steganography Histogram");

	//encode_lsb_alt(img, read_file("test.txt"));
	//cout << decode_lsb_alt(img);

	img = encode_dct(img, "test");
	//cout << decode_dct(img);

	//showHistogram(img, "Post-Steganography Histogram");

	cvShowImage("Image", img);
	cvWaitKey();

	//system("pause");
	return EXIT_SUCCESS;
}