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
	unsigned char* originalImagePixels = new unsigned char[originalImagePixmap.height() * originalImagePixmap.width() * 3];

	// Need to convert to QImage to access pixels
	QImage originalImage = originalImagePixmap.toImage();

	// loop over each pixel
	for (int y = 0; y < originalImage.height(); ++y) {
		for (int x = 0; x < originalImage.width(); ++x) {
			QRgb pixel = originalImage.pixel(QPoint(x, y));

			originalImagePixels[3 * (x + y * originalImage.width()) + 0] = qRed(pixel);
			originalImagePixels[3 * (x + y * originalImage.width()) + 1] = qGreen(pixel);
			originalImagePixels[3 * (x + y * originalImage.width()) + 2] = qBlue(pixel);
		}
	}
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