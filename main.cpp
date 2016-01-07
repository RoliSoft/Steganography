#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/algorithm/string.hpp>
#include "format.h"
#include "helpers.hpp"
#include "lsb.hpp"
#include "lsb_alt.hpp"
#include "dct.hpp"
#include "dwt.hpp"
#include "tlv.hpp"

#if _WIN32
	#include <conio.h>
#else
	#define _getche getchar
#endif

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
		 << "  Similarity: " << setprecision(3) << color << Format::Bold << accuracy << "%" << Format::Normal << Format::Default << endl << endl
		 << "  Input:"       << endl << endl << Format::White << Format::Bold << original  << Format::Normal << Format::Default << endl << endl
		 << "  Extracted:"   << endl << endl << Format::White << Format::Bold << extracted << Format::Normal << Format::Default << endl;
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
	auto img = imread("test/img_small.png");

	show_image(img, "Original");

	auto input  = read_file("test/test.txt");
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
	auto img = imread("test/img_small.png");

	show_image(img, "Original");

	auto input  = read_file("test/test.txt");
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
	auto img = imread("test/lena.jpg");

	show_image(img, "Original");

	auto input  = read_file("test/test.txt");
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
	auto img = imread("test/lena.jpg");

	show_image(img, "Original");

	auto input = read_file("test/test.txt");
	auto stego = encode_dct(img,   input, STORE_FULL, 0);
	     stego = encode_dct(stego, input, STORE_FULL, 1);
		 stego = encode_dct(stego, input, STORE_FULL, 2);

	imwrite("test/lena_dct.jpg", stego, vector<int> { CV_IMWRITE_JPEG_QUALITY, 80 });
	stego = imread("test/lena_dct.jpg");

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
	auto img = imread("test/lena.jpg");

	show_image(img, "Original");

	auto input  = read_file("test/test.txt");
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
	auto img = imread("test/lena.jpg");

	show_image(img, "Original");

	auto input = read_file("test/test.txt");
	auto stego = encode_dwt(img,   input, STORE_FULL, 0);
	     stego = encode_dwt(stego, input, STORE_FULL, 1);
		 stego = encode_dwt(stego, input, STORE_FULL, 2);

	imwrite("test/lena_dwt.jpg", stego, vector<int> { CV_IMWRITE_JPEG_QUALITY, 90 });
	stego = imread("test/lena_dwt.jpg");

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
 * Prompts the user for a selection from the available options.
 *
 * \param opts Available options.
 *
 * \return Option selected by user.
 */
char get_selection(const string& opts)
{
	cout << "  Selection: " << Format::Green << Format::Bold;

	char sel;
	while ((sel = _getche()))
	{
		if (opts.find(sel) != string::npos)
		{
			break;
		}

		cout << Format::Normal << Format::Default << endl
			 << "  Selection: " << Format::Green << Format::Bold;
	}

	cout << Format::Normal << Format::Default << endl;

	return sel;
}

/*!
 * Builds a menu for the user and prompts a selection.
 *
 * \param title Title of the menu.
 * \param opts Available options.
 * \param actions Options which perform an action.
 *
 * \return Option selected by user.
 */
char show_menu(const string& title, const vector<pair<char, string>>& opts, const string& actions = string())
{
	cout << endl << "  " << title << ":" << endl;

	string sels;

	for (auto& p : opts)
	{
		sels += p.first;
		auto color = p.first == 'b' ? Format::White : actions.find(p.first) != string::npos ? Format::Cyan : Format::Green;
		cout << "    [" << color << Format::Bold << p.first << Format::Normal << Format::Default << "] " << Format::Bold << p.second << Format::Normal << endl;
	}

	cout << endl;

	return get_selection(sels);
}

/*!
 * Translates a STORE_* constant into a string.
 *
 * \param store Constant value to translate.
 *
 * \return Translated value.
 */
string store_to_string(int store)
{
	switch (store)
	{
	case STORE_ONCE:   return "Store Once, Leave Rest Random";
	case STORE_FULL:   return "Store Once, Fill Rest w/ Zeros";
	case STORE_REPEAT: return "Store as Indefinitely Repeating";
	default:           return "Unknown Mode " + to_string(store);
	}
}

/*!
 * Prompts the user to select a storage mode.
 *
 * \param store Storage variable to manipulate.
 */
void select_store(int& store)
{
	switch (show_menu("Storage Mode", {
		{ 'o', "Store Once, Leave Rest Random" },
		{ 'z', "Store Once, Fill Rest w/ Zeros" },
		{ 'r', "Store as Indefinitely Repeating" },
		{ 'b', "Back to Main Menu" }
	}))
	{
	case 'o': store = STORE_ONCE;   break;
	case 'z': store = STORE_FULL;   break;
	case 'r': store = STORE_REPEAT; break;
	}
}

/*!
 * Translates the channel parameter value into a string.
 *
 * \param channel Parameter value to translate.
 *
 * \return Translated value.
 */
string channel_to_string(int channel)
{
	switch (channel)
	{
	case 0:  return "Encode All Channels";
	case 1:  return "Encode Red Channel";
	case 2:  return "Encode Green Channel";
	case 3:  return "Encode Blue Channel";
	default: return "Unknown Mode " + to_string(channel);
	}
}

/*!
 * Prompts the user to select a channel.
 *
 * \param channel Channel variable to manipulate.
 */
void select_channel(int& channel)
{
	switch (show_menu("Storage Mode", {
		{ 'a', "Encode All Channels" },
		{ 'r', "Encode Red Channel" },
		{ 'g', "Encode Green Channel" },
		{ 'k', "Encode Blue Channel" },
		{ 'b', "Back to Main Menu" }
	}))
	{
	case 'a': channel = 0; break;
	case 'r': channel = 1; break;
	case 'g': channel = 2; break;
	case 'k': channel = 3; break;
	}
}

/*!
 * Prompts the user to provide a string value.
 *
 * \param title Name of the variable.
 * \param value Variable to manipulate.
 * \param checkFile Check if value is an existing file.
 */
void prompt_string(const string& title, string& value, bool checkFile = false)
{
	cout << endl << "  " << title << ": " << Format::Green << Format::Bold;

	string str;
	getline(cin, str);
	trim(str);

	cout << Format::Normal << Format::Default;

	if (str.length() > 0)
	{
		if (checkFile)
		{
			ifstream fs(str);
			if (fs.good())
			{
				value = str;
			}
			else
			{
				cerr << "    " << Format::Red << Format::Bold << "Error:" << Format::Normal << Format::Default << " Specified file '" << str << "' does not exist." << endl;
			}
		}
		else
		{
			value = str;
		}
	}
}

/*!
 * Prompts the user to provide an integer value.
 *
 * \param title Name of the variable.
 * \param value Variable to manipulate.
 * \param min Lower limit of the value.
 * \param max Upper limit of the value.
 */
void prompt_int(const string& title, int& value, int min = INT_MIN, int max = INT_MAX)
{
	cout << endl << "  " << title << (min != INT_MIN && max != INT_MAX ? " [" + to_string(min) + "-" + to_string(max) + "]" : "") << ": " << Format::Green << Format::Bold;

	string str;
	getline(cin, str);
	trim(str);

	cout << Format::Normal << Format::Default;

	if (str.length() > 0)
	{
		auto num = atoi(str.c_str());

		if (num <= max && num >= min)
		{
			value = num;
		}
		else
		{
			cerr << "    " << Format::Red << Format::Bold << "Error:" << Format::Normal << Format::Default << " Specified value " << num << " out of range " << to_string(min) << " - " << to_string(max) << "." << endl;
		}
	}
}

/*!
 * Prompts the user to provide a double value.
 *
 * \param title Name of the variable.
 * \param value Variable to manipulate.
 * \param min Lower limit of the value.
 * \param max Upper limit of the value.
 */
void prompt_double(const string& title, double& value, double min = DBL_MIN, double max = DBL_MAX)
{
	cout << endl << "  " << title << (min != DBL_MIN && max != DBL_MAX ? " [" + to_string(min) + "-" + to_string(max) + "]" : "") << ": " << Format::Green << Format::Bold;

	string str;
	getline(cin, str);
	trim(str);

	cout << Format::Normal << Format::Default;

	if (str.length() > 0)
	{
		auto num = atof(str.c_str());

		if (num <= max && num >= min)
		{
			value = num;
		}
		else
		{
			cerr << "    " << Format::Red << Format::Bold << "Error:" << Format::Normal << Format::Default << " Specified value " << num << " out of range " << to_string(min) << " - " << to_string(max) << "." << endl;
		}
	}
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

	cout << Format::Yellow << Format::Bold << endl;
	cout << "       ____ __                                                     __       " << endl;
	cout << "      / __// /_ ___  ___ _ ___ _ ___  ___  ___ _ ____ ___ _ ___   / /  __ __" << endl;
	cout << "     _\\ \\ / __// -_)/ _ `// _ `// _ \\/ _ \\/ _ `// __// _ `// _ \\ / _ \\/ // /" << endl;
	cout << "    /___/ \\__/ \\__/ \\_, / \\_,_//_//_/\\___/\\_, //_/   \\_,_// .__//_//_/\\_, / " << endl;
	cout << "                   /___/                 /___/           /_/         /___/  " << endl;
	cout << endl;
	cout << "                  " << Format::Red << Format::Bold << "https" << Format::Normal << Format::Red << "://" << Format::Bold << "github.com" << Format::Normal << Format::Red << "/" << Format::Bold << "RoliSoft" << Format::Normal << Format::Red << "/" << Format::Bold << "Steganography" << Format::Normal << Format::Default << endl;

main:
	switch (show_menu("Available Options", {
		{ 'i', "Image Processing" },
		{ 'v', "Video Processing" },
		{ 't', "Unit Tests" },
		{ 'x', "Exit" }
	}, "x"))
	{
	case 'i':
	{
		switch (show_menu("Technique to Use", {
			{ 'l', "Least Significant Bit" },
			{ 'c', "Discrete Cosine Transformation" },
			{ 'w', "Discrete Wavelet Transformation" },
			{ 'b', "Back to Main Menu" }
		}))
		{
		case 'l':
		{
			string input = "test/img.png";
			auto store = 1, channel = 0;

		mnlsb:
			switch (show_menu("LSB Configuration", {
				{ 'i', "Input File:    " + input },
				{ 's', "Storage Mode:  " + store_to_string(store) },
				{ 'c', "Channel Usage: " + channel_to_string(channel) },
				{ 'a', "Perform Steganography" },
				{ 'b', "Back to Main Menu" }
			}, "a"))
			{
			case 'i':
				prompt_string("Input File", input, true);
				goto mnlsb;

			case 's':
				select_store(store);
				goto mnlsb;

			case 'c':
				select_channel(channel);
				goto mnlsb;

			case 'a':
				//do_lsb(input, store, channel);
				goto main;

			case 'b':
				goto main;
			}
		}
		break;

		case 'c':
		{
			string input = "test/lena.jpg";
			auto store = 1, channel = 0, persistence = 30;

		mndct:
			switch (show_menu("DCT Configuration", {
				{ 'i', "Input File:    " + input },
				{ 's', "Storage Mode:  " + store_to_string(store) },
				{ 'c', "Channel Usage: " + channel_to_string(channel) },
				{ 'p', "Persistence:   " + to_string(persistence) + "%" },
				{ 'a', "Perform Steganography" },
				{ 'b', "Back to Main Menu" }
			}, "a"))
			{
			case 'i':
				prompt_string("Input File", input, true);
				goto mndct;

			case 's':
				select_store(store);
				goto mndct;

			case 'c':
				select_channel(channel);
				goto mndct;

			case 'p':
				prompt_int("Persistence Percentage", persistence, 0, 100);
				goto mndct;

			case 'a':
				//do_dct(input, store, channel, persistence);
				goto main;

			case 'b':
				goto main;
			}
		}
		break;

		case 'w':
		{
			string input = "test/lena.jpg";
			auto store = 1, channel = 0;
			auto alpha = 0.1;

		mndwt:
			switch (show_menu("DWT Configuration", {
				{ 'i', "Input File:    " + input },
				{ 's', "Storage Mode:  " + store_to_string(store) },
				{ 'c', "Channel Usage: " + channel_to_string(channel) },
				{ 'p', "Intensity:     " + to_string(alpha) },
				{ 'a', "Perform Steganography" },
				{ 'b', "Back to Main Menu" }
			}, "a"))
			{
			case 'i':
				prompt_string("Input File", input, true);
				goto mndwt;

			case 's':
				select_store(store);
				goto mndwt;

			case 'c':
				select_channel(channel);
				goto mndwt;

			case 'p':
				prompt_double("Intensity Value", alpha, 0.01, 1);
				goto mndwt;

			case 'a':
				//do_dwt(input, store, channel, alpha);
				goto main;

			case 'b':
				goto main;
			}
		}
		break;

		case 'b':
			goto main;
		}
	}
	break;

	case 'v':
		break;

	case 't':
	{
		switch (show_menu("Available Unit Tests", {
			{ '1', "Least Significant Bit -- All Channels" },
			{ '2', "Least Significant Bit -- Alternating Channels" },
			{ '3', "Discrete Cosine Transformation -- Single Channel" },
			{ '4', "Discrete Cosine Transformation -- All Channels w/ JPEG Compression" },
			{ '5', "Discrete Wavelet Transformation -- Single Channel" },
			{ '6', "Discrete Wavelet Transformation -- All Channels w/ JPEG Compression" },
			{ 'b', "Back to Main Menu" }
		}, "123456"))
		{
		case '1':
			test_lsb();
			cvWaitKey();
			break;
		case '2':
			test_lsb_alt();
			cvWaitKey();
			break;
		case '3':
			test_dct();
			cvWaitKey();
			break;
		case '4':
			test_dct_multi();
			cvWaitKey();
			break;
		case '5':
			test_dwt();
			cvWaitKey();
			break;
		case '6':
			test_dwt_multi();
			cvWaitKey();
			break;
		case 'b':
			goto main;
		}
	}
	break;

	case 'x':
		return EXIT_SUCCESS;
	}
	
	return EXIT_SUCCESS;
}
