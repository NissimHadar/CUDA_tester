#include "ImageEditor.h"

#include <QFileDialog>
#include <QMessageBox>

ImageEditor::ImageEditor(std::unique_ptr<QLabel> originalImage, std::unique_ptr<QLabel> editedImage) {
	this->originalImage = std::move(originalImage);
	this->editedImage = std::move(editedImage);

	this->originalImage->setScaledContents(true);
	this->editedImage->setScaledContents(true);
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

	QPixmap imagePixmap = QPixmap(filename);
	originalImage->setPixmap(imagePixmap);
}