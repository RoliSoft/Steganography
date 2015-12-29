#pragma once
#include <string>

/*!
* Encapsulates the specified input into TLV format.
* In order to hide the data from easy fingerprinting, this is not a true
* tag-length-value format, it instead uses a method where the tag is
* derived from the length, tag = ~length.
*
* \param text Input to be encapsulated.
*
* \return Encapsulated text.
*/
std::string encode_tlv(const std::string& text)
{
	auto size = int(text.length());
	auto xize = ~size;

	return std::string(reinterpret_cast<char*>(&size), sizeof(int)) + std::string(reinterpret_cast<char*>(&xize), sizeof(int)) + text;
}

/*!
 * Extracts the text encapsulated within the obfuscated/pseudo-TLV format. 
 *
 * \param text Input to be processed.
 *
 * \return Extracted text or original string on failure.
 */
std::string decode_tlv(const std::string& text)
{
	auto size = *reinterpret_cast<const int*>(text.c_str());
	auto xize = *reinterpret_cast<const int*>(text.c_str() + sizeof(int));

	if (xize != ~size)
	{
		return text;
	}

	return text.substr(sizeof(int) * 2, size);
}
