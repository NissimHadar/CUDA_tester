#include "CUDA_tester.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[]) {
	QApplication a(argc, argv);
	CUDA_tester w;
	w.show();
	return a.exec();
}
