
#ifndef COMPLEX_CUH
#define COMPLEX_CUH

#include <cuda_runtime.h>
#include <math.h>

#define HALF_PI 1.57079632679f
#define PI 3.14159265359f
#define TWO_PI 6.28318530718f

struct complex {
	float re, im;

	__host__ __device__ complex(float x = 0.0f, float y = 0.0f);

	__host__ __device__ void operator=(const complex& z);
	__host__ __device__ void operator=(const float& a);

	__host__ __device__ complex operator-() const;

	__host__ __device__ complex operator+(const complex& z) const;
	__host__ __device__ complex operator+(const float& a) const;

	__host__ __device__ void operator+=(const complex& z);
	__host__ __device__ void operator+=(const float& a);

	__host__ __device__ complex operator-(const complex& z) const;
	__host__ __device__ complex operator-(const float& a) const;

	__host__ __device__ void operator-=(const complex& z);
	__host__ __device__ void operator-=(const float& a);

	__host__ __device__ complex operator*(const complex& z) const;
	__host__ __device__ complex operator*(const float& a) const;

	__host__ __device__ void operator*=(const complex& z);
	__host__ __device__ void operator*=(const float& a);

	__host__ __device__ complex operator/(const complex& z) const;
	__host__ __device__ complex operator/(const float& a) const;

	__host__ __device__ void operator/=(const complex& z);
	__host__ __device__ void operator/=(const float& a);

	__host__ __device__ complex operator!() const; // complex conjugate
};

__host__ __device__ void print_complex(const complex& z);

__host__ __device__ complex operator+(const float& a, const complex& z);
__host__ __device__ complex operator-(const float& a, const complex& z);
__host__ __device__ complex operator*(const float& a, const complex& z);
__host__ __device__ complex operator/(const float& a, const complex& z);

__host__ __device__ float re(const complex& z);
__host__ __device__ float im(const complex& z);
__host__ __device__ float arg(const complex& z); // ]-pi, pi[
__host__ __device__ float mod(const complex& z);
__host__ __device__ float mod2(const complex& z);

#define ZERO complex(0.0f, 0.0f) // 0 + 0i
#define ONE complex(1.0f, 0.0f) // 1 + 0i
#define I complex(0.0f, 1.0f) // 0 + 1i
#define IS_ZERO(z) ((z.re) == 0.0f && (z.im) == 0.0f)
#define SO2(theta) complex(cosf(theta), sinf(theta)) // cos(theta) + i*sin(theta)

__host__ __device__ complex exp(const complex& z);
__host__ __device__ complex log(const complex& z);
__host__ __device__ complex pow(const complex& z, const complex& w);
__host__ __device__ complex pow(const complex& z, const float& alpha);
__host__ __device__ complex pow(const float& a, const complex& w);
__host__ __device__ complex sin(const complex& z);
__host__ __device__ complex cos(const complex& z);
__host__ __device__ complex tan(const complex& z);
__host__ __device__ complex sinh(const complex& z);
__host__ __device__ complex cosh(const complex& z);
__host__ __device__ complex tanh(const complex& z);
__host__ __device__ complex sinc(const complex& z);
__host__ __device__ complex tanc(const complex& z);
__host__ __device__ complex nthroot(const complex& z, unsigned int n);
__host__ __device__ complex zeta(const complex& z);


struct domain {
	float min_re, max_re, min_im, max_im;
	size_t N; // kleinste discretisatie
	size_t width, height; // breedte en hoogte in pixels

	__host__ domain(float min_Re, float max_Re, float min_Im, float max_Im, size_t n);
};

__host__ void print_domain(const domain& D, const char* name) {
	printf("Domain %s:\n", name);
	printf("[%f , %f] x [%f , %f]\n", D.min_re, D.max_re, D.min_im, D.max_im);
	printf("N = %zu\n", D.N);
	printf("Canvas size: %zu x %zu\n", D.width, D.height);
}

#define PRINT_DOMAIN(D) print_domain(D, #D)


#ifdef COMPLEX_IMPLEMENTATION

__host__ __device__ void print_complex(const complex& z) {
	printf("%f + i*%f\n", z.re, z.im);
}

__host__ __device__ complex::complex(float x, float y) {
	re = x;
	im = y;
}

__host__ __device__ void complex::operator=(const complex& z) {
	re = z.re;
	im = z.im;
}
__host__ __device__ void complex::operator=(const float& a) {
	re = a;
	im = 0.0f;
}

__host__ __device__ complex complex::operator-() const {
	return complex(-re, -im);
}

__host__ __device__ complex complex::operator+(const complex& z) const {
	return complex(re + z.re, im + z.im);
}
__host__ __device__ complex complex::operator+(const float& a) const {
	return complex(re + a, im);
}

__host__ __device__ void complex::operator+=(const complex& z) {
	re += z.re;
	im += z.im;
}
__host__ __device__ void complex::operator+=(const float& a) {
	re += a;
}

__host__ __device__ complex complex::operator-(const complex& z) const {
	return complex(re - z.re, im - z.im);
}
__host__ __device__ complex complex::operator-(const float& a) const {
	return complex(re - a, im);
}

__host__ __device__ void complex::operator-=(const complex& z) {
	re -= z.re;
	im -= z.im;
}
__host__ __device__ void complex::operator-=(const float& a) {
	re -= a;
}

__host__ __device__ complex complex::operator*(const complex& z) const {
	// (x+iy)(u+iv) = (xu-yv) + i(xv+yu)
	return complex(re * z.re - im * z.im, re * z.im + im * z.re);
}
__host__ __device__ complex complex::operator*(const float& a) const {
	return complex(a * re, a * im);
}

__host__ __device__ void complex::operator*=(const complex& z) {
	re = re * z.re - im * z.im;
	im = re * z.im + im * z.re;
}
__host__ __device__ void complex::operator*=(const float& a) {
	re *= a;
	im *= a;
}

__host__ __device__ complex complex::operator/(const complex& z) const {
	// assert(z.re != 0.0f && z.im != 0.0f);
	// assert(!IS_ZERO(z));
	// (x+iy)/(u+iv) = (x+iy)(u-iv)/(u^2+v^2) = ((xu+yv) + i(yu - xv))/(u^2+v^2)
	float n = z.re * z.re + z.im * z.im;
	return complex((re * z.re + im * z.im) / n, (im * z.re - re * z.im) / n);
}
__host__ __device__ complex complex::operator/(const float& a) const {
	// assert(a != 0.0f);
	return complex(re / a, im / a);
}

__host__ __device__ void complex::operator/=(const complex& z) {
	// assert(z.re != 0.0f && z.im != 0.0f);
	// assert(!IS_ZERO(z));
	float n = z.re * z.re + z.im * z.im;
	re = (re * z.re + im * z.im) / n;
	im = (im * z.re - re * z.im) / n;
}
__host__ __device__ void complex::operator/=(const float& a) {
	// assert(a != 0.0f);
	re /= a;
	im /= a;
}

__host__ __device__ complex complex::operator!() const {
	return complex(re, -im);
}

__host__ __device__ complex operator+(const float& a, const complex& z) {
	return complex(z.re + a, z.im);
}
__host__ __device__ complex operator-(const float& a, const complex& z) {
	return complex(z.re - a, z.im);
}
__host__ __device__ complex operator*(const float& a, const complex& z) {
	return complex(a * z.re, a * z.im);
}
__host__ __device__ complex operator/(const float& a, const complex& z) {
	float n = z.re * z.re + z.im * z.im;
	return complex(a * z.re / n, -a * z.im / n);
}

__host__ __device__ float re(const complex& z) {
	return z.re;
}
__host__ __device__ float im(const complex& z) {
	return z.im;
}
__host__ __device__ float arg(const complex& z) {
	return atan2(z.im, z.re);
}
__host__ __device__ float mod(const complex& z) {
	return sqrtf(z.re * z.re + z.im * z.im);
}
__host__ __device__ float mod2(const complex& z) {
	return z.re * z.re + z.im * z.im;
}

__host__ __device__ complex exp(const complex& z) {
	// e^z = e^re(z)*e^(i*im(z))
	return expf(z.re) * SO2(z.im);
}
__host__ __device__ complex log(const complex& z) {
	// assert(!IS_ZERO(z));
	// log(z) = log|z| + i*arg(z)
	return complex(0.5f * logf(mod2(z)), arg(z));
}
__host__ __device__ complex pow(const complex& z, const complex& w) {
	// z^w = e^(w*log(z))
	if (IS_ZERO(w)) return ONE;
	if (IS_ZERO(z)) return ZERO;
	return exp(w * log(z));
}
__host__ __device__ complex pow(const complex& z, const float& alpha) {
	if (alpha == 0.0f) return ONE;
	if (IS_ZERO(z)) return ZERO;
	return powf(mod2(z), alpha / 2.0f) * SO2(alpha * arg(z));
}
__host__ __device__ complex pow(const float& a, const complex& w) {
	return powf(a, w.re) * SO2(w.im * logf(a));
}
__host__ __device__ complex sin(const complex& z) {
	// 1/i = -i
	return -0.5f * I * (exp(I * z) - exp(-I * z));
}
__host__ __device__ complex cos(const complex& z) {
	return 0.5f * (exp(I * z) + exp(-I * z));
}
__host__ __device__ complex tan(const complex& z) {
	return -I * (exp(I * z) - exp(-I * z)) / (exp(I * z) + exp(-I * z));
}
__host__ __device__ complex sinh(const complex& z) {
	return 0.5f * (exp(z) - exp(-z));
}
__host__ __device__ complex cosh(const complex& z) {
	return 0.5f * (exp(z) + exp(-z));
}
__host__ __device__ complex tanh(const complex& z) {
	return (exp(z) - exp(-z)) / (exp(z) + exp(-z));
}
__host__ __device__ complex sinc(const complex& z) {
	if (IS_ZERO(z)) return ONE;
	return sin(z) / z;
}
__host__ __device__ complex tanc(const complex& z) {
	if (IS_ZERO(z)) return ONE;
	return tan(z) / z;
}
__host__ __device__ complex nthroot(const complex& z, unsigned int n) {
	// assert(n > 0);
	return pow(z, 1.0f / n);
}
__host__ __device__ complex zeta(const complex& z) {
	if (0.0f < z.re && z.re < 1.0f) {
		// kritieke streep
	}
	return complex();
}

__host__ domain::domain(float min_Re, float max_Re, float min_Im, float max_Im, size_t n) {
	min_re = min_Re;
	max_re = max_Re;
	min_im = min_Im;
	max_im = max_Im;
	N = n;

	float dxf = max_re - min_re;
	float dyf = max_im - min_im;
	if (dxf < dyf) {
		width = N;
		height = (size_t)(dyf * N / dxf);
	}
	else {
		height = N;
		width = (size_t)(dxf * N / dyf);
	}
}

#endif // COMPLEX_IMPLEMENTATION

#endif // COMPLEX_CUH
