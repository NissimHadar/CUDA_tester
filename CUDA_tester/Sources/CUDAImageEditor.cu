#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAImageEditor.cuh"


__global__
void removeBlue(const unsigned int width, const unsigned char* const inputPixels, unsigned char* const outputPixels) {
	// Set third byte to 0
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	unsigned int byteIndex = 3 * (x + width * y);

	// Use weighted values
	int colour = (0.2125 * inputPixels[byteIndex + 0]) + (0.7154 * inputPixels[byteIndex + 1]) + (0.0721 * inputPixels[byteIndex + 2]);

	outputPixels[byteIndex + 0] = colour;
	outputPixels[byteIndex + 1] = colour;
	outputPixels[byteIndex + 2] = colour;
}

void CUDAImageEditor::convertToMonochrome(const unsigned int height, const unsigned int width, const unsigned char* const h_inputPixels, unsigned char* const h_outputPixels) {
	const unsigned int BUFFER_SIZE{ height * width * 3 };

	// Put pixel buffer in device memory
	unsigned char* d_inputPixels;
	unsigned char* d_outputPixels;
	cudaMalloc(&d_inputPixels, BUFFER_SIZE);
	cudaMalloc(&d_outputPixels, BUFFER_SIZE);

	cudaMemcpy(d_inputPixels, h_inputPixels, BUFFER_SIZE, cudaMemcpyHostToDevice);

	// Blocks will be 8x8 threads
	dim3 threadsPerBlock(8, 8);

	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	removeBlue<<< numBlocks, threadsPerBlock >>>(width, d_inputPixels, d_outputPixels);

	cudaMemcpy((void *)h_outputPixels, d_outputPixels, BUFFER_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_inputPixels);
	cudaFree(d_outputPixels);
}