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

### Discrete Wavelet Transformation

Uses discrete wavelet transformation (specifically [Haar](https://en.wikipedia.org/wiki/Haar_wavelet)) to hide data in the diagonal filter of a channel within an image.

This method can also survive in lossy formats, but its efficiency is as not as good as DCT's, therefore this should mainly be used for watermarking purposes. A 90% JPEG compression will not degrade the data at all, however lower compression values require a bigger persistance during the hiding process, which starts to visibly deteriorate the image.

## Utilities

In order to aid data hiding with the implemented methods, some helper functions have been implemented.

### Encapsulation

This function encapsulates the data to be hidden in a TLV (Tag-Length-Value) packet.

In order to hide the data from easy fingerprinting, this is not a true tag-length-value format, it instead uses a method where the tag is derived from the length, `tag = ~length`.

### Reconstruction

In order to facilitate the use of multiple channels with multiple methods, there is a function to compare the output of each method per channel and try to reconstruct the original message by picking the most frequent character for each index within the specified method outputs.

This way, minor to major errors, depending on the quality of the outputs, can be corrected. In order for the algorithm to properly function, it requires at least 3 strings from different channels or methods.

![Screenshot](https://i.imgur.com/509HbZN.jpg)