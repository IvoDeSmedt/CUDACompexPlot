
#ifndef EXPORT_CUH
#define EXPORT_CUH

#include "include/stb_image_write.h"

#define COMP 4

int writePNG(const char* filename, unsigned int width, unsigned int height, const void* data);


#ifdef EXPORT_IMPLEMENTATION

int writePNG(const char* filename, unsigned int width, unsigned int height, const void* data) {

	// comp = 3 voor RGB
	// 4 voor RGBA?
	return stbi_write_png(filename, width, height, COMP, data, COMP * width);
}

#endif // EXPORT_IMPLEMENTATION

#endif // EXPORT_CUH
