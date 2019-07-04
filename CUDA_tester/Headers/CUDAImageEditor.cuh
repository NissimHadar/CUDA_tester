#ifndef _CUDAImageEditor_h__
#define _CUDAImageEditor_h__

class CUDAImageEditor {
public:
	void convertToMonochrome(const unsigned int height, const unsigned int width, const unsigned char* const h_inputPixels, unsigned char* const h_outputPixels);
};

#endif
