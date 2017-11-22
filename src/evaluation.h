#ifndef WRITE_H
#define WRITE_H

#include <fstream>
#include <sstream>
#include "bitmap.hpp"
#include "calculation.h"



void visualizeParameters(const std::vector<std::vector<double>> & conditionalProbabilities) {

	for (unsigned int c = 0; c < 10; c++) {

		std::vector<unsigned char> classFs(784);

		for (unsigned int f = 0; f < 784; f++) {

			double p = conditionalProbabilities[c][f];
			uint8_t v = 255 * p;
			classFs[f] = (unsigned char)v;
		}

		std::stringstream ss;
		ss << "../output/digit" << c <<".bmp";
		std::cout << "Writing to " << ss.str() << std::endl;
		Bitmap::writeBitmap(classFs, 28, 28, ss.str(), false);
	}
}



void writeNetwork(const std::vector<double> & priorProbabilities, const std::vector<std::vector<double>> & conditionalProbabilities) {

	std::cout << "Writing to ../output/network.txt" << std::endl;
	std::ofstream ofs("../output/network.txt");

	for (unsigned int f = 0; f < 784; f++) {
		ofs << conditionalProbabilities[0][f] << std::endl;
	}

	for (unsigned int f = 0; f < 784; f++) {
		ofs << conditionalProbabilities[1][f] << std::endl;
	}

	for (unsigned int c = 0; c < 10; c++) {
		ofs << priorProbabilities[c] << std::endl;
	}

	ofs.close();
}



void writeClassificationSummary(const std::vector<unsigned char> & testLabels, const std::vector<unsigned int> & predictedLabels, const double & accuracy) {

	std::cout << "Writing to ../output/classification-summary.txt" << std::endl;
	std::ofstream ofs("../output/classification-summary.txt");

	for (unsigned int r = 0; r < 10; r++) {

		for (unsigned int c = 0; c < 10; c++) {

			unsigned int counter = 0;

			for (unsigned int i = 0; i < static_cast<unsigned int>(testLabels.size()); i++) {

				if (static_cast<unsigned int>(testLabels[i]) == r && predictedLabels[i] == c) {
					counter++;
				}
			}

			ofs << counter << ",\t";
		}

		ofs << std::endl;
	}

	ofs << accuracy << std::endl;
	ofs.close();
}



#endif