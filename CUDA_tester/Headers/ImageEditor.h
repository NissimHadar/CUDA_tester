#ifndef __ImageEditor_h__
#define __ImageEditor_h__

#include <QLabel>

class ImageEditor {
public:
	ImageEditor(std::unique_ptr<QLabel> originalImage, std::unique_ptr<QLabel> editedImage);

	void editImage();
	void selectImage();

private:
	std::unique_ptr<QLabel> originalImage;
	std::unique_ptr<QLabel> editedImage;

	QPixmap originalImagePixmap;
};

#endif
