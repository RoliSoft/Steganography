# Steganography

The scope of this classroom assignment within the image processing course is to try and efficiently hide within the images, preferably without noticable visual degradation and some resistance to compression or manipulation.

## Methods

Some implemented methods more are better suited for watermarking, while others can conceal full files within.

### Least Significant Bit

Works by manipulating the least significant bit of each pixel in order to hide a message. This method can hide the most amount of data, but requires a lossless format, such as PNG or BMP.

Any compression or manipulation with 3rd-party tools will lead to significant data degradation to the carried message.

### Discrete Cosine Transformation

Uses [DCT](http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct) to hide data in the coefficients of a channel within an image.

This method can survive an 80% JPEG compression while keeping the data completely intact and without introducing any significant visual degradation to the image during the hiding process. Better survival rates can be achieved, by using multiple channels and a bigger persistence value, however visual degradation may start appearing depending on the image being used.

This technique also works for video steganography. While video compression can introduce a heavy data loss in regards to steganographic artifacts, a high-enough-bitrate H.264-encoded video (such as the supplied test file) can be processed and re-encoded, resulting in the same file size, same image quality, and reproducible hidden content.

Further information regarding this method is available in [Lin, Yih-Kai. "A data hiding scheme based upon DCT coefficient modification." _Computer Standards & Interfaces_ 36.5 (2014): 855-862.](http://ms12.voip.edu.tw/~paul/Papper/Steganography/DCT/A_data_hiding_scheme_based_upon_DCT_coefficient_modification.pdf)

### Discrete Wavelet Transformation

Uses discrete wavelet transformation (specifically [Haar](https://en.wikipedia.org/wiki/Haar_wavelet)) to hide data in the diagonal filter of a channel within an image.

This method can also survive in lossy formats, but its efficiency is as not as good as DCT's, therefore this should mainly be used for watermarking purposes. A 90% JPEG compression will not degrade the data at all, however lower compression values require a bigger persistance during the hiding process, which starts to visibly deteriorate the image.

Further information regarding this method is available in [Kumar, Sushil, and S. K. Muttoo. "Data Hiding Techniques Based on Wavelet-like Transform and Complex Wavelet Transforms." _2010 International Symposium on Intelligence Information Processing and Trusted Computing_. IEEE, 2010.](https://www.academia.edu/3632247/Data_Hiding_techniques_Based_On_Wavelet-like_transform_and_Complex_Wavelet_Transforms)

## Utilities

In order to aid data hiding with the implemented methods, some helper functions have been implemented.

### Encapsulation

This function encapsulates the data to be hidden in a TLV (Tag-Length-Value) packet.

In order to hide the data from easy fingerprinting, this is not a true tag-length-value format, it instead uses a method where the tag is derived from the length, `tag = ~length`.

### Reconstruction

In order to facilitate the use of multiple channels with multiple methods, there is a function to compare the output of each method per channel and try to reconstruct the original message by picking the most frequent character for each index within the specified method outputs.

This way, minor to major errors, depending on the quality of the outputs, can be corrected. In order for the algorithm to properly function, it requires at least 3 strings from different channels or methods.

## Building

The project was originally developed under Visual Studio 2015 and linked against OpenCV 3.1 x64, however the application should be compilable under any modern operating system, as Windows-specific calls and structs were aliased to their POSIX equivalents and handled accordingly.

Under Windows, the `opencv_world310[d].dll` file is the only required dependency during runtime for the image processing features. For the video processing features `opencv_ffmpeg310[_64].dll` will also be required, and optionally an encoder/decoder library to handle various video formats. To process H.264 videos, including the supplied test video, the `openh264-1.4.0-win[64|32]msvc.dll` file can be downloaded from [cisco/openh264](https://github.com/cisco/openh264/releases).

![Screenshot](https://i.imgur.com/509HbZN.jpg)