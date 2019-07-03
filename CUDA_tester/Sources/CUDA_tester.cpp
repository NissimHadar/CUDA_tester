#include "CUDA_tester.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"

CUDA_tester::CUDA_tester(QWidget *parent) : QMainWindow(parent) {
	ui.setupUi(this);

	imageEditor = std::make_unique<ImageEditor>(std::unique_ptr<QLabel>(ui.labelOriginalImage), std::unique_ptr<QLabel>(ui.labelEditedImage));

	// Check number of CUDA devices
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		ui.textEdit->setText(QString("cudaGetDeviceCount returned ") + static_cast<int>(error_id) + " -> " + cudaGetErrorString(error_id) + ".\n");
	} else if (deviceCount == 0) {
		ui.textEdit->setText(QString("There are no available device(s) that support CUDA.\n"));
	} else {
		if (deviceCount == 1) {
			ui.textEdit->setText("Detected  one CUDA Capable device.\n");
		} else {
			ui.textEdit->setText(QString("Detected ") + QString::number(deviceCount) + "CUDA Capable devices.\n");
		}
	}

	// Check capabilities of each device
	for (int device = 0; device < deviceCount; ++device) {

		// Device type
		cudaSetDevice(device);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		ui.textEdit->append(QString("Device ") + QString::number(device) + ": " + deviceProp.name);

		// Driver version
		int driverVersion;
		int runtimeVersion;
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		ui.textEdit->append(QString("  CUDA Driver Version, Runtime Version: ") 
			+ QString::number(driverVersion  / 1000) + "." + QString::number((driverVersion  % 100) / 10) + ", "
			+ QString::number(runtimeVersion / 1000) + "." + QString::number((runtimeVersion % 100) / 10)
		);

		// CUDA capability version
		ui.textEdit->append(QString("  CUDA Capability Version: ")
			+ QString::number(deviceProp.major) + "." + QString::number(deviceProp.minor)
		);

		// Global memory
		ui.textEdit->append(QString("  Total amount of global memory: ") +
			QString::number(static_cast<float>(deviceProp.totalGlobalMem / (1 << 20))) + " MB"
		);

		// L2 memory
		if (deviceProp.l2CacheSize) {
			ui.textEdit->append(QString("  L2 Cache Size:") + QString::number(deviceProp.l2CacheSize / (1 << 20)) + " MB");
		}

		// Number of multi-processors and cores per multi-processor
		ui.textEdit->append(QString("  ") +
			QString::number(deviceProp.multiProcessorCount) + " Multi-processors and " +
			QString::number(_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)) + " CUDA cores/MP -> " +
			QString::number(_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount) + " CUDA cores"
		);

		// Warp size
		ui.textEdit->append(QString("  Warp size: ") + QString::number(deviceProp.warpSize));

		// Maximum number of threads per multiprocessor
		ui.textEdit->append(QString("  Maximum number of threads per multiprocessor: ") + QString::number(deviceProp.maxThreadsPerMultiProcessor));

		// Maximum number of threads per block
		ui.textEdit->append(QString("  Maximum number of threads per block: ") + QString::number(deviceProp.maxThreadsPerBlock));

		//  Max dimension size of a thread block
		ui.textEdit->append(QString("  Max dimension size of a thread block (x,y,z):  ") +
			QString::number(deviceProp.maxThreadsDim[0]) + ", " + QString::number(deviceProp.maxThreadsDim[1]) + ", " + QString::number(deviceProp.maxThreadsDim[2])
		);

		// Max dimension size of a grid
		ui.textEdit->append(QString("  Max dimension size of a grid  (x,y,z): ") +
			QString::number(deviceProp.maxGridSize[0]) + ", " + QString::number(deviceProp.maxGridSize[1]) + ", " + QString::number(deviceProp.maxGridSize[2])
		);
	}
}

void CUDA_tester::on_pushButtonEditImage_clicked() {
	imageEditor->editImage();
}

void CUDA_tester::on_pushButtonSelectImage_clicked() {
	imageEditor->selectImage();
}

void CUDA_tester::on_pushButtonClose_clicked() {
	exit(0);
}

