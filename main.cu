
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <time.h>

#define COMPLEX_IMPLEMENTATION
#include "complex.cuh"

#define COLOR_IMPLEMENTATION
#include "color.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define EXPORT_IMPLEMENTATION
#include "export.cuh"

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<<grid, block>>>
#else
#define KERNEL_ARGS2(grid, block)
#endif

// grootte en fijnheid van het raster, een iteratie-functie en een kleurenfunctie
// geef maar 1 parameter door voor de fijnheid (N), want het doel zijn vierkantige pixels
cudaError_t iter_make_frac(float min_re, float max_re, float min_im, float max_im, size_t N, complex(*iter)(complex), size_t MAXIT, color(*assign)(complex));

__global__ void plot_initialise(unsigned int max_x, unsigned int max_y) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= max_x) || (y >= max_y)) {
        printf("index (%i, %i) out of bounds (%i, %i)\n", x, y, max_x, max_y);
    }
}

typedef complex(*complex_valued)(const complex&);
typedef color(*for_drawing)(const complex&, plotType);

__device__ complex id(const complex& z) {
    return z;
}
__device__ complex iter(const complex& z) {
    return z - pow(exp(z) * log(z), z);
}

__device__ complex_valued F = id;
__device__ complex_valued G = iter;
__device__ for_drawing A = plot_color;

__global__ void plot_function(unsigned char* dev_framebuffer, domain D, complex_valued f, for_drawing assign, plotType type) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // if ((x >= D.width) || (y >= D.height)) return;

    unsigned int pixel_index = y * D.width + x;
    complex z(
        D.min_re + (float)x * (D.max_re - D.min_re) / D.width,
        D.min_im + (float)(D.height - y) * (D.max_im - D.min_im) / D.height
    );
    color C = A(F(z), type);
    // color C;
    dev_framebuffer[COMP * pixel_index + 0] = C.r;
    dev_framebuffer[COMP * pixel_index + 1] = C.g;
    dev_framebuffer[COMP * pixel_index + 2] = C.b;
    dev_framebuffer[COMP * pixel_index + 3] = C.a;
}

__global__ void plot_iter(unsigned char* dev_framebuffer, domain D, size_t ITER, plotType type, bool rotate) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // if ((x >= D.width) || (y >= D.height)) return;

    unsigned int pixel_index = y * D.width + x;
    complex z;
    if (rotate) {
        z = complex(
            D.min_im + (float)(D.height - y) * (D.max_im - D.min_im) / D.height,
            D.min_re + (float)x * (D.max_re - D.min_re) / D.width
        );
    }
    else {
        z = complex(
            D.min_re + (float)x * (D.max_re - D.min_re) / D.width,
            D.min_im + (float)(D.height - y) * (D.max_im - D.min_im) / D.height
        );
    }
    for (size_t IT = 0; IT < ITER; IT++) z = G(z);
    color C = A(z, type);
    // color C;
    dev_framebuffer[COMP * pixel_index + 0] = C.r;
    dev_framebuffer[COMP * pixel_index + 1] = C.g;
    dev_framebuffer[COMP * pixel_index + 2] = C.b;
    dev_framebuffer[COMP * pixel_index + 3] = C.a;
}

// plot draaien?
// fracDS:
// 1. z -= tan(z^2)/z^2
// 2. z -= tan^2(z)
// 3. z -= tan(z^3)/z^3
// 4. z -= tan(z^z)/z^z
// 5. z -= i^(1/z)
// 6. z -= i^(1/z) met verschilfunctie als parameter
// 7. z -= z^(1/z)
int main() {

    float min_re = -20, max_re = 20;
    float min_im = -15, max_im = 15;

    size_t N = 1 << 10;

    size_t ITER = 1e2;


    // [CPU] initialisatie
    domain D(min_re, max_re, min_im, max_im, N);
    PRINT_DOMAIN(D);
    size_t fb_size = D.width * D.height * COMP * sizeof(unsigned char);
    unsigned char* framebuffer = (unsigned char*)malloc(fb_size);
    
    // [GPU] initialisatie
    unsigned char* dev_framebuffer;
    cudaError_t status = cudaMalloc((void**)&dev_framebuffer, fb_size), sync;
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed. Error string:\n");
        cudaGetErrorString(status);
        goto error;
    }
    
    int tx = 8;
    int ty = 8;
    dim3 threads(tx, ty);
    dim3 blocks(D.width / tx, D.height / ty);

    clock_t start, stop;

    // [GPU] plotten
    start = clock();
    plot_initialise KERNEL_ARGS2(blocks, threads) (D.width, D.height);
    // plot_function KERNEL_ARGS2(blocks, threads) (dev_framebuffer, D, &id, &plot_color, argColor);
    plot_iter KERNEL_ARGS2(blocks, threads) (dev_framebuffer, D, ITER, argGrayscale, true);
    status = cudaGetLastError();
    sync = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "plot failed. Last error string: %s\n", cudaGetErrorString(status));
        goto error;
    }
    if (sync != cudaSuccess) {
        fprintf(stderr, "synchronisation failed. Last error string: %s\n", cudaGetErrorString(sync));
        goto error;
    }
    stop = clock();
    float timerSec = ((float)(stop - start)) / CLOCKS_PER_SEC;
    printf("Initialiseren en renderen van de afbeelding duurde %f seconden.\n", timerSec);

    // [GPU -> CPU] de data kopiëren
    start = clock();
    status = cudaMemcpy(framebuffer, dev_framebuffer, fb_size, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy device -> host failed. Error string: %s\n", cudaGetErrorString(status));
        goto error;
    }
    stop = clock();
    timerSec = ((float)(stop - start)) / CLOCKS_PER_SEC;
    printf("Afbeelding van de GPU naar de CPU kopiëren duurde %f seconden.\n", timerSec);

    // [CPU] afbeelding schrijven
    start = clock();
    const char* file = "iter_C.png";
    // flipData(D.dx, D.dy, framebuffer);
    int ret = writePNG(file, D.width, D.height, framebuffer);
    stop = clock();
    timerSec = ((float)(stop - start)) / CLOCKS_PER_SEC;
    printf("Afbeelding %s naar PNG schrijven duurde %f seconden.\n", file, timerSec);

    // afbeelding weergeven met Windows
    system("iter_C.png");

error:
    cudaFree(dev_framebuffer);
    free(framebuffer);

    cudaDeviceReset();

    return 0;
}
