#ifndef CALCULATION_H
#define CALCULATION_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cfloat>



// traing step 1.a: calculate the prior frequencies for each class c
std::vector<unsigned int> calculatePriorFrequencies(const std::vector<unsigned char> & trainLabels) {

	std::cout << "Calculating Prior Frequencies..." << std::endl;
	std::vector<unsigned int> priorFrequencies;
	unsigned int totalNumberImages = static_cast<unsigned int>(trainLabels.size());     // total number of images in the training set

	// count the prior frequencies for each class c
	for (unsigned int c = 0; c < 10; c++) {

		unsigned int priorFrequency = 0;	// initialize the counter of images of this class

		// iterate through the training labels to count the prior frequency of this class
        for (unsigned int i = 0; i < totalNumberImages; i++) {

            // increment the counter of this image
            if (static_cast<unsigned int>(trainLabels[i]) == c) {
                priorFrequency++;
            }
        }

        priorFrequencies.push_back(priorFrequency);
        // std::cout << "\tClass " << c << ": " << priorFrequency << std::endl;
	}

	// std::cout << std::endl;
	return priorFrequencies;
}



// traing step 1.b: determine the prior probabilities for each class c
std::vector<double> calculatePriorProbabilities(const std::vector<unsigned char> & trainLabels, const std::vector<unsigned int> & priorFrequencies) {

	std::cout << "Calculating Prior Probabilities..." << std::endl;
    std::vector<double> priorProbabilities;                                             
    unsigned int totalNumberImages = static_cast<unsigned int>(trainLabels.size());     // total number of images in the training set

    // determine the prior probabilities for each class c
    for (unsigned int c = 0; c < 10; c++) {

        double priorProbability = static_cast<double>(priorFrequencies[c]) / static_cast<double>(totalNumberImages);
        priorProbabilities.push_back(priorProbability);
        // std::cout << "\tClass " << c << ": " << priorProbability << std::endl;
    }

    // std::cout << std::endl;
    return priorProbabilities;
}


// training step 2.a: calculate the conditional frequencies that pixel (Fj == 1) given (C == c)
std::vector<std::vector<unsigned int>> calculateConditionalFrequencies(const std::vector<std::vector<unsigned char>> & trainImages, const std::vector<unsigned char> & trainLabels) {

	std::cout << "Calculating Conditional Frequencies..." << std::endl;
	std::vector<std::vector<unsigned int>> conditionalFrequencies;
	unsigned int numberFeatures = static_cast<unsigned int>(trainImages[0].size());		// number of pixels in each training image
    unsigned int totalNumberImages = static_cast<unsigned int>(trainLabels.size());		// total number of images in the training set

    // given that it is an image of digit c
    for (unsigned int c = 0; c < 10; c++) {

    	// std::cout << "\tClass " << c << ":" << std::endl;
    	std::vector<unsigned int> conditionalFrequency;		// the conditional probability given (C == c)

        // count the frequency of each feature (Fj == 1)
        for (unsigned int f = 0; f < numberFeatures; f++) {

        	unsigned int numberWhitePixels = 0;		// initialize the counter of images of digit c where pixel Fj is white

        	// iterate through the training images and labels to count the number of images of digit c where pixel Fj is white
            for (unsigned int i = 0; i < totalNumberImages; i++) {

                if (static_cast<unsigned int>(trainLabels[i]) == c && static_cast<unsigned int>(trainImages[i][f]) == 1) {
                    numberWhitePixels++;
                }
            }

            conditionalFrequency.push_back(numberWhitePixels);
            // std::cout << "\t\tFeature " << f << ": \t" << numberWhitePixels << std::endl;
        }

        conditionalFrequencies.push_back(conditionalFrequency);
    }

    // std::cout << std::endl;
    return conditionalFrequencies;
}



// training step 2.b: determine the conditional probabilities that pixel (Fj == 1) given (C == c)
std::vector<std::vector<double>> calculateConditionalProbabilities(const std::vector<std::vector<unsigned int>> & conditionalFrequencies, const std::vector<unsigned int> & priorFrequencies) {

	std::cout << "Calculating Conditional Probabilities with Laplace Smoothing..." << std::endl;
    std::vector<std::vector<double>> conditionalProbabilities;
    unsigned int numberFeatures = static_cast<unsigned int>(conditionalFrequencies[0].size());		// number of pixels in each training image

    // given that it is an image of digit c
    for (unsigned int c = 0; c < 10; c++) {

    	// std::cout << "\tClass " << c << ":" << std::endl;
        std::vector<double> conditionalProbability;     // the conditional probability given (C == c)

        // calculate the conditional probability of each feature (Fj == 1)
        for (unsigned int f = 0; f < numberFeatures; f++) {

        	// Laplace Smoothing
        	double probability = static_cast<double>(conditionalFrequencies[c][f] + 1) / static_cast<double>(priorFrequencies[c]);
        	conditionalProbability.push_back(probability);
        	// std::cout << "\t\tFeature " << f << ": \t" << probability << std::endl;
        }

        conditionalProbabilities.push_back(conditionalProbability);
    }

    // std::cout << std::endl;
    return conditionalProbabilities;
}



// testing step 1.a: compute the probabilities that one testing image belongs to each class c
std::vector<double> calculateTestingProbabilities(const std::vector<unsigned char> & testImage, const std::vector<double> & priorProbabilities, const std::vector<std::vector<double>> & conditionalProbabilities) {

	// std::cout << "Calculating Testing Probabilities..." << std::endl;
	std::vector<double> testingProbabilities;
	unsigned int numberFeatures = static_cast<unsigned int>(conditionalProbabilities[0].size());	// number of pixels in each training image

	// iterate through each class c
	for (unsigned int c = 0; c < 10; c++) {

		double testingProbability = log(priorProbabilities[c]);		// initialize the testing probability for this class

		// iterate through each feature Fj
		for (unsigned int f = 0; f < numberFeatures; f++) {

			// current feature is white (Fj == 1)
			if (static_cast<unsigned int>(testImage[f]) == 1) {
				testingProbability += log(conditionalProbabilities[c][f]);
			}

			// current feature is black (Fj == 0)
			else {
				testingProbability += log(1.0 - conditionalProbabilities[c][f]);
			}
		}

		testingProbabilities.push_back(testingProbability);
		// std::cout << "\tClass " << c << ": " << testingProbability << std::endl;
	}

	// std::cout << std::endl;
	return testingProbabilities;
}


std::vector<unsigned int> calculatePredictedLabels(const std::vector<std::vector<unsigned char>> & testImages, const std::vector<double> & priorProbabilities, const std::vector<std::vector<double>> & conditionalProbabilities) {

	std::cout << "Calculating Predicted Labels..." << std::endl;
	std::vector<unsigned int> predictedLabels;
	unsigned int totalNumberImages = static_cast<unsigned int>(testImages.size());		// total number of images in the testing set

	// iterate through each test image
	for (unsigned int i = 0; i < totalNumberImages; i++) {

		std::vector<double> testingProbabilities = calculateTestingProbabilities(testImages[i], priorProbabilities, conditionalProbabilities);
		unsigned int maximumTestingLabel = -1;
		double maximumTestingProbability = -DBL_MAX;

		for (unsigned int c = 0; c < static_cast<unsigned int>(testingProbabilities.size()); c++) {

			if (testingProbabilities[c] >= maximumTestingProbability) {

				maximumTestingLabel = c;
				maximumTestingProbability = testingProbabilities[c];
			}
		}

		unsigned int predictedLabel = maximumTestingLabel;
		predictedLabels.push_back(predictedLabel);
		// std::cout << "\tPredicted Label " << i << ": \t" << predictedLabel << std::endl;
	}

	// std::cout << std::endl;
	return predictedLabels;
}



double calculateAccuracy(const std::vector<unsigned char> & testLabels, const std::vector<unsigned int> & predictedLabels) {

	unsigned int accurate = 0;
	unsigned int total = static_cast<unsigned int>(testLabels.size()) ;
	std::cout << "Calculating Accuracy for " << total << " Test Images..." << std::endl;

	for (unsigned int i = 0; i < total; i++) {

		// std::cout << "\tTest Image " << i << ": \t" << static_cast<unsigned int>(testLabels[i]) << " vs " << predictedLabels[i] << std::endl;
		if (static_cast<unsigned int>(testLabels[i]) == predictedLabels[i]) {
			accurate++;
		}
	}

	// std::cout << "Accurate #: " << accurate << std::endl;
	double accurary = (double)accurate / (double)total;
	std::cout << "Accuracy %: " << accurary << std::endl;

	return accurary;
}



#endif