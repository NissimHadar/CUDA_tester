#include "ImageEditor.h"
#include "CUDAImageEditor.cuh"

#include <QFileDialog>
#include <QMessageBox>

ImageEditor::ImageEditor(std::unique_ptr<QLabel> originalImage, std::unique_ptr<QLabel> editedImage) {
	this->originalImage = std::move(originalImage);
	this->editedImage = std::move(editedImage);

	this->originalImage->setScaledContents(true);
	this->editedImage->setScaledContents(true);
}

void ImageEditor::editImage() {
	// Need to convert to QImage to access pixels
	QImage originalImage = originalImagePixmap.toImage();

	// Pixels sent to device
	unsigned char* originalImagePixels = new unsigned char[originalImagePixmap.height() * originalImagePixmap.width() * 3];

	// Pixels received from device
	unsigned char* editedImagePixels = new unsigned char[originalImagePixmap.height() * originalImagePixmap.width() * 3];

	// loop over each pixel
	int height{ originalImage.height() };
	int width{ originalImage.width() };

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			QRgb pixel = originalImage.pixel(QPoint(x, y));

			originalImagePixels[3 * (x + y * width) + 0] = qRed(pixel);
			originalImagePixels[3 * (x + y * width) + 1] = qGreen(pixel);
			originalImagePixels[3 * (x + y * width) + 2] = qBlue(pixel);
		}
	}

	CUDAImageEditor cudaImageEditor;
	cudaImageEditor.convertToMonochrome(height, width, originalImagePixels, editedImagePixels);

	QPixmap pixmap = QPixmap::fromImage(QImage(editedImagePixels, width, height, QImage::Format_RGB888));
	editedImage->setPixmap(pixmap);
}

void ImageEditor::selectImage() {
		QString filename = QFileDialog::getOpenFileName(
		nullptr,
		"Select an image",
		QDir::currentPath(),
		"Image (*.png *.jpg *.jpeg *.bmp *.gif))"
	);

	if (filename == QString::null) {
		return;
	}

	originalImagePixmap = QPixmap(filename);
	originalImage->setPixmap(originalImagePixmap);
}