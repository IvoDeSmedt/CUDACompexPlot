
#ifndef COLOR_CUH
#define COLOR_CUH

#include <cuda_runtime.h>
#include "complex.cuh"

enum plotType {
	argColor_modAlpha = 0,
	argColor = 1,
	argGrayscale_modAlpha = 2,
	argGrayscale = 3,
	argPartialGrayscale_modAlpha = 4,
	argPartialGrayscale = 5
};

struct color {
	unsigned char r, g, b, a;

	__host__ __device__ color(unsigned char R = 0, unsigned char G = 0, unsigned char B = 0, unsigned char A = 255);
};

__host__ __device__ color RGB_disk(float theta);

__host__ __device__ unsigned char GF256(float a, float min = 0.0f, float max = 1.0f);

__host__ __device__ color plot_color(const complex& z, plotType type);


#ifdef COLOR_IMPLEMENTATION

__host__ __device__ color::color(unsigned char R, unsigned char G, unsigned char B, unsigned char A) {
	r = R;
	g = G;
	b = B;
	a = A;
}

__host__ __device__ float emod(float f, float g) {
	float r = f;
	while (r > g) r -= g;
	return r;
}

__host__ __device__ color RGB_disk(float theta) {
	unsigned int m = (int)(3.0f * theta / PI);
	unsigned char p = GF256(emod(theta, PI / 3), 0.0f, PI / 3);
	if (m == 0) return color(255, p, 0);
	if (m == 1) return color(255 - p, 255, 0);
	if (m == 2) return color(0, 255, p);
	if (m == 3) return color(0, 255 - p, 255);
	if (m == 4) return color(p, 0, 255);
	if (m == 5) return color(255, 0, 255 - p);
	return color();
}

__host__ __device__ color grayscale_disk(float theta) {
	unsigned char t = GF256(theta, -PI, PI);
	return color(t, t, t);
}

__host__ __device__ color partial_grayscale_disk(float theta, float min, float max) {
	unsigned char t = GF256(theta, min, max);
	return color(t, t, t);
}


__host__ __device__ unsigned char GF256(float a, float min, float max) {
	return (int)(255 * (a - min) / (max - min));
}

__host__ __device__ color plot_color(const complex& z, plotType type) {
	color C;
	if (type == argColor_modAlpha) {
		C = RGB_disk(arg(-z) + PI); // kleur (argument + pi)
		C.a = (int)floorf(255 * expf(1.0f - mod(z))); // ondoorzichtigheid (modulus)
	}
	else if (type == argColor) {
		C = RGB_disk(arg(-z) + PI);
		C.a = 255;
	}
	else if (type == argGrayscale_modAlpha) {
		C = grayscale_disk(arg(z));
		C.a = (int)floorf(255 * expf(1.0f - mod(z)));
	}
	else if (type == argGrayscale) {
		C = grayscale_disk(arg(z));
		C.a = 255;
	}
	else if (type == argPartialGrayscale_modAlpha) {
		C = partial_grayscale_disk(arg(z), -PI, 0);
		C.a = (int)floorf(255 * expf(1.0f - mod(z)));
	}
	else if (type == argPartialGrayscale) {
		C = partial_grayscale_disk(arg(z), -PI, 0);
		C.a = 255;
	}
	return C;
}

#endif // COLOR_IMPLEMENTATION

#endif // IMAGE_CUH
