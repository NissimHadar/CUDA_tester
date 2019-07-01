#ifndef __CUDA_tester_h__
#define __CUDA_tester_h__

#include <QtWidgets/QMainWindow>
#include "ui_CUDA_tester.h"

#include "ImageEditor.h"

class CUDA_tester : public QMainWindow
{
	Q_OBJECT

public:
	CUDA_tester(QWidget *parent = Q_NULLPTR);

private:
	Ui::CUDA_testerClass ui;

	std::unique_ptr<ImageEditor> imageEditor;

private slots:
	void on_pushButtonSelectImage_clicked();
	void on_pushButtonClose_clicked();
};

#endif
