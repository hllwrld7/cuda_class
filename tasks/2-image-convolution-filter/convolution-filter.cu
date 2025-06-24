#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <vector>

using namespace std;

typedef vector<vector<float>> Matrix;

const int BYTES_PER_PIXEL = 3; /// red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{ 0x4D42 };  // 'BM'
    uint32_t fileSize;
    uint16_t reserved1{ 0 };
    uint16_t reserved2{ 0 };
    uint32_t offsetData{ 54 + 1024 };  // Header + color palette (256 * 4)
};

struct DIBHeader {
    uint32_t size{ 40 };
    int32_t width;
    int32_t height;
    uint16_t planes{ 1 };
    uint16_t bitCount{ 8 };  // 8-bit grayscale
    uint32_t compression{ 0 };
    uint32_t sizeImage;
    int32_t xPixelsPerMeter{ 2835 };
    int32_t yPixelsPerMeter{ 2835 };
    uint32_t colorsUsed{ 256 };
    uint32_t colorsImportant{ 256 };
};
#pragma pack(pop)

using Matrix = std::vector<std::vector<float>>;

// Generate a horizontal gradient (left to right)
std::vector<std::vector<uint8_t>> generateGrayscaleImage(int width, int height) {
    std::vector<std::vector<uint8_t>> image(height, std::vector<uint8_t>(width));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image[y][x] = static_cast<uint8_t>((x * 255) / (width - 1));
    return image;
}

// Save 8-bit grayscale BMP
void saveBMP(const std::string& filename, const std::vector<std::vector<uint8_t>>& image) {
    int height = image.size();
    int width = image[0].size();
    int rowSize = (width + 3) & ~3; // padded to 4-byte boundary
    int imageSize = rowSize * height;

    BMPHeader bmp;
    DIBHeader dib;
    dib.width = width;
    dib.height = height;
    dib.sizeImage = imageSize;
    bmp.fileSize = bmp.offsetData + imageSize;

    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Could not open file for writing.");

    out.write(reinterpret_cast<char*>(&bmp), sizeof(bmp));
    out.write(reinterpret_cast<char*>(&dib), sizeof(dib));

    // Write grayscale color palette (256 entries)
    for (int i = 0; i < 256; ++i) {
        uint8_t gray[4] = { static_cast<uint8_t>(i), static_cast<uint8_t>(i), static_cast<uint8_t>(i), 0 };
        out.write(reinterpret_cast<char*>(gray), 4);
    }

    // Write image data (bottom-up)
    std::vector<uint8_t> row(rowSize, 0);
    for (int y = height - 1; y >= 0; --y) {
        std::copy(image[y].begin(), image[y].end(), row.begin());
        out.write(reinterpret_cast<char*>(row.data()), rowSize);
    }

    out.close();
}

// Read 8-bit grayscale BMP into float matrix
Matrix readBMPtoFloatMatrix(const std::string& filename, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file");

    BMPHeader bmpHeader;
    DIBHeader dibHeader;

    file.read(reinterpret_cast<char*>(&bmpHeader), sizeof(bmpHeader));
    file.read(reinterpret_cast<char*>(&dibHeader), sizeof(dibHeader));

    if (bmpHeader.fileType != 0x4D42)
        throw std::runtime_error("Not a BMP file");

    if (dibHeader.bitCount != 8)
        throw std::runtime_error("Only 8-bit BMP supported");

    width = dibHeader.width;
    height = dibHeader.height;
    int rowSize = (width + 3) & ~3;

    // Skip color palette
    file.seekg(bmpHeader.offsetData, std::ios::beg);

    Matrix image(height, std::vector<float>(width));
    std::vector<uint8_t> rowData(rowSize);

    for (int y = height - 1; y >= 0; --y) {
        file.read(reinterpret_cast<char*>(rowData.data()), rowSize);
        for (int x = 0; x < width; ++x)
            image[y][x] = static_cast<float>(rowData[x]);  // Keep 0–255 range
    }

    return image;
}

void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName);
unsigned char* createBitmapFileHeader(int height, int stride);
unsigned char* createBitmapInfoHeader(int height, int width);

void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName)
{
    int widthInBytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = { 0, 0, 0 };
    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes)+paddingSize;

    FILE* imageFile = fopen(imageFileName, "wb");

    unsigned char* fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

    int i;
    for (i = 0; i < height; i++) {
        fwrite(image + (i * widthInBytes), BYTES_PER_PIXEL, width, imageFile);
        fwrite(padding, 1, paddingSize, imageFile);
    }

    fclose(imageFile);
}

unsigned char* createBitmapFileHeader(int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] = {
        0,0,     /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
    };

    fileHeader[0] = (unsigned char)('B');
    fileHeader[1] = (unsigned char)('M');
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

unsigned char* createBitmapInfoHeader(int height, int width)
{
    static unsigned char infoHeader[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0,     /// number of color planes
        0,0,     /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };

    infoHeader[0] = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[4] = (unsigned char)(width);
    infoHeader[5] = (unsigned char)(width >> 8);
    infoHeader[6] = (unsigned char)(width >> 16);
    infoHeader[7] = (unsigned char)(width >> 24);
    infoHeader[8] = (unsigned char)(height);
    infoHeader[9] = (unsigned char)(height >> 8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

    return infoHeader;
}

// Example filter kernels students can implement:

// Box blur (3x3)
float boxBlur3x3[9] = {
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

// Gaussian blur (5x5)
float gaussianBlur5x5[25] = {
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f};

// Sobel edge detection (horizontal)
float sobelX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1};

// Sobel edge detection (vertical)
float sobelY[9] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1};

// Sharpen filter
float sharpen[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0};

// Utility to check for CUDA errors
#define CHECK_CUDA_ERROR(call)                                        \
    {                                                                 \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess)                                       \
        {                                                             \
            fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                    cudaGetErrorString(err), __LINE__, __FILE__);     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    }

// Structure to hold image data
struct Image
{
    vector<vector<float>> data;
    int width;
    int height;
    int channels; // 1 for grayscale, 3 for RGB, 4 for RGBA

    Image(int w, int h, int c, vector<vector<float>> d) : width(w), height(h), channels(c), data(d){}

};

Matrix generateGaussianKernel(int size, float sigma) {
    Matrix kernel(size, std::vector<float>(size));
    float sum = 0.0f;

    int half = size / 2;
    float sigma2 = 2.0f * sigma * sigma;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            float value = std::exp(-(i * i + j * j) / sigma2);
            kernel[i + half][j + half] = value;
            sum += value;
        }
    }

    // Normalize the kernel so that the sum is 1
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            kernel[i][j] /= sum;

    return kernel;
}

void saveFloatMatrixAsBMP(const std::string& filename, const Matrix& mat) {
    int height = mat.size();
    int width = mat[0].size();
    int rowSize = (width + 3) & ~3;
    int imageSize = rowSize * height;

    // Find min and max for normalization
    float minVal = mat[0][0], maxVal = mat[0][0];
    for (const auto& row : mat)
        for (float val : row) {
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }

    // Normalize to [0, 255]
    std::vector<std::vector<uint8_t>> image(height, std::vector<uint8_t>(width));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image[y][x] = static_cast<uint8_t>(
                255.0f * (mat[y][x] - minVal) / (maxVal - minVal + 1e-5f)
                );

    // BMP and DIB headers
    BMPHeader bmp;
    DIBHeader dib;
    bmp.offsetData = 54 + 1024;
    dib.width = width;
    dib.height = height;
    dib.sizeImage = imageSize;
    bmp.fileSize = bmp.offsetData + imageSize;

    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to write BMP file.");

    out.write(reinterpret_cast<char*>(&bmp), sizeof(bmp));
    out.write(reinterpret_cast<char*>(&dib), sizeof(dib));

    // Write grayscale color palette
    for (int i = 0; i < 256; ++i) {
        uint8_t entry[4] = { static_cast<uint8_t>(i), static_cast<uint8_t>(i), static_cast<uint8_t>(i), 0 };
        out.write(reinterpret_cast<char*>(entry), 4);
    }

    // Write image (bottom-up)
    std::vector<uint8_t> row(rowSize);
    for (int y = height - 1; y >= 0; --y) {
        std::copy(image[y].begin(), image[y].end(), row.begin());
        out.write(reinterpret_cast<char*>(row.data()), rowSize);
    }

    out.close();
}

// CPU implementation of 2D convolution
Matrix convolutionCPU(const Matrix& input, const Matrix& kernel)
{
    int inputHeight = input.size();
    int inputWidth = input[0].size();
    int kernelHeight = kernel.size();
    int kernelWidth = kernel[0].size();

    int outputHeight = inputHeight - kernelHeight + 1;
    int outputWidth = inputWidth - kernelWidth + 1;

    Matrix output(outputHeight, std::vector<float>(outputWidth, 0.0f));

    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernelHeight; ++ki) {
                for (int kj = 0; kj < kernelWidth; ++kj) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }

    return output;
}

// Naive GPU implementation - each thread computes one output pixel
__global__ void convolutionKernelNaive(const float* input, float* output, int width, int height,
    const float* kernel, int kernelSize)
{
    // TODO: Implement naive GPU version of convolution
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernelHalf = kernelSize / 2;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int ky = -kernelHalf; ky <= kernelHalf; ++ky) {
        for (int kx = -kernelHalf; kx <= kernelHalf; ++kx) {
            int ix = x + kx;
            int iy = y + ky;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                float val = input[iy * width + ix];
                float kval = kernel[(ky + kernelHalf) * kernelSize + (kx + kernelHalf)];
                sum += val * kval;
            }
        }
    }
    output[y * width + x] = sum;
}

// Shared memory implementation
__global__ void convolutionKernelShared(const float* input, float* output,
    int width, int height,
    const float* kernel, int kSize)
{
    // TODO: Implement shared memory version of convolution
    extern __shared__ float tile[];

    const int kHalf = kSize / 2;
    const int tileSize = blockDim.x + 2 * kHalf;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int sharedX = tx + kHalf;
    int sharedY = ty + kHalf;

    // Load tile into shared memory
    if (x < width && y < height)
        tile[sharedY * tileSize + sharedX] = input[y * width + x];
    else
        tile[sharedY * tileSize + sharedX] = 0.0f;

    // Load halo (only needed by threads at the tile edges)
    // Horizontal halo
    if (tx < kHalf) {
        int left = x - kHalf;
        tile[sharedY * tileSize + tx] = (left >= 0) ? input[y * width + left] : 0.0f;

        int right = x + blockDim.x;
        if (right < width)
            tile[sharedY * tileSize + (sharedX + blockDim.x)] = input[y * width + right];
        else
            tile[sharedY * tileSize + (sharedX + blockDim.x)] = 0.0f;
    }

    // Vertical halo
    if (ty < kHalf) {
        int top = y - kHalf;
        tile[ty * tileSize + sharedX] = (top >= 0) ? input[top * width + x] : 0.0f;

        int bottom = y + blockDim.y;
        if (bottom < height)
            tile[(sharedY + blockDim.y) * tileSize + sharedX] = input[bottom * width + x];
        else
            tile[(sharedY + blockDim.y) * tileSize + sharedX] = 0.0f;
    }

    __syncthreads();

    // Now perform convolution
    float sum = 0.0f;
    if (x < width && y < height) {
        for (int ky = 0; ky < kSize; ++ky)
            for (int kx = 0; kx < kSize; ++kx) {
                float kval = kernel[ky * kSize + kx];
                float val = tile[(sharedY + ky - kHalf) * tileSize + (sharedX + kx - kHalf)];
                sum += val * kval;
            }
        output[y * width + x] = sum;
    }
}

// Constants for filter definitions
__constant__ float d_filter[81]; // Max 9x9 filter

// Main function to compare implementations
int main(int argc, char **argv)
{
    // TODO: Load or generate an image
    int width = 512, height = 512;

    // Step 1: Generate and save image
    auto image = generateGrayscaleImage(width, height);
    saveBMP("output.bmp", image);
    std::cout << "Saved image: output.bmp\n";

    // Step 2: Read image back into float matrix
    int readWidth, readHeight;
    Matrix matrix = readBMPtoFloatMatrix("output.bmp", readWidth, readHeight);

    // TODO: Define convolution filters (e.g., blur, sharpen, edge detection)

    auto filter = generateGaussianKernel(5, 1.0f);
    
    // TODO: Implement timing utilities

    clock_t start, end;
    double cpu_time_used, gpu_time_used, gpu_time_used0;
    
    start = clock();

    // TODO: Run CPU implementation

    auto cpuOutput = convolutionCPU(matrix, filter);
    saveFloatMatrixAsBMP("convolved_output.bmp", cpuOutput);

    end = clock();

    // Calculate time taken in seconds
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("cpu Time taken: %f seconds\n", cpu_time_used);

    // TODO: Run GPU implementations
    start = clock();

    int imgSize = width * height;
    std::vector<float> h_input(imgSize), h_output(imgSize);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            h_input[y * width + x] = matrix[y][x];

    std::vector<float> h_kernel(25);
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 5; ++x)
            h_kernel[y * 5 + x] = filter[y][x];

    float* d_input, * d_output, * d_kernel;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, imgSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, imgSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, 25 * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), imgSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel.data(), 25 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    convolutionKernelNaive<<<grid, block>>>(d_input, d_output, width, height, d_kernel, 5);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaMemcpy(h_output.data(), d_output, imgSize * sizeof(float), cudaMemcpyDeviceToHost);

    Matrix gpuOutput(height, std::vector<float>(width));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            gpuOutput[y][x] = h_output[y * width + x];

    saveFloatMatrixAsBMP("convoled_output_gpu_naive.bmp", gpuOutput);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    end = clock();
    gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("gpu Time taken (naive): %f seconds\n", gpu_time_used);

    start = clock();
    dim3 grid1((width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);
    size_t sharedMemSize = (block.x + 5 - 1) * (block.y + 5 - 1) * sizeof(float);

    convolutionKernelShared<<<grid1, block, sharedMemSize>>>(d_input, d_output, width, height, d_kernel, 5);
    cudaDeviceSynchronize();


    end = clock();
    gpu_time_used0 = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("gpu Time taken (shared memory): %f seconds\n", gpu_time_used0);

    // TODO: Compare results and performance

    return 0;
}