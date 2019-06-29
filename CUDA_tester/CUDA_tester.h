#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_CUDA_tester.h"

class CUDA_tester : public QMainWindow
{
	Q_OBJECT

public:
	CUDA_tester(QWidget *parent = Q_NULLPTR);

private:
	Ui::CUDA_testerClass ui;

private slots:
	void on_pushButtonClose_clicked();
};
