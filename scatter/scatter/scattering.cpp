#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>
#include <ctime>
#include <iomanip>

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#include "medium.h"

//#define nextEvent

int i, j;

int main(int argc, char *argv[]) {

	// Thread protect random number generate
	int nworkers = omp_get_num_procs();
    omp_set_num_threads(nworkers);

	std::vector<Sampler> samplers;
	samplers.resize(nworkers);
	{
		std::random_device rd;
		for (int i = 0; i < nworkers; ++i) samplers[i].init(rd());
	}

	// UI
	std::ifstream file(argv[1]);
	const int N_Sample = std::atoi(argv[2]);
	const int h_sigT = std::atoi(argv[3]);
	const int w_sigT = std::atoi(argv[4]);
	const double y = std::atof(argv[5]);
	const double x = std::atof(argv[6]);
	const double receiptorSize = std::atof(argv[7]);
	const int block = std::atoi(argv[8]);
	std::ifstream fileAlbedo(argv[9]);

	// read albedo list
	Eigen::VectorXd albedoList(block);
	for (i = 0; i < 1; ++i) {
		std::string line;
		std::getline(fileAlbedo, line);

		std::stringstream iss(line);
		for (j = 0; j < block; ++j) {
			std::string val;
			std::getline(iss, val, ',');
			std::stringstream convertor(val);
			convertor >> albedoList[j];
		}
	}
	
	// read sigT from csv
	Eigen::MatrixXd sigT(h_sigT, w_sigT);
	for (i = 0; i < h_sigT; ++i) {
		std::string line;
		std::getline(file, line);

		std::stringstream iss(line);
		for (j = 0; j < w_sigT; ++j) {
			std::string val;
			std::getline(iss, val, ',');
			std::stringstream convertor(val);
			convertor >> sigT(i,j);
		}
	}

	// define output reflectance
	double reflectanceTotal = 0.0;
	Eigen::VectorXd reflectanceBlock = Eigen::VectorXd::Zero(block);
	Eigen::MatrixXd reflectance = Eigen::MatrixXd::Zero(nworkers, block);

	double reflectanceTotal2 = 0.0;
	Eigen::VectorXd reflectanceBlock2 = Eigen::VectorXd::Zero(block);
	Eigen::MatrixXd reflectance2 = Eigen::MatrixXd::Zero(nworkers, block);

	double reflectanceStderrTotal = 0.0;
	Eigen::VectorXd reflectanceStderrBlock = Eigen::VectorXd::Zero(block);

	// define medium 
	//Medium<2> *med;
	//if (sigT.maxCoeff() - sigT.minCoeff() < 0.0000001) {
	//	med = new HomogeneousMedium<2>(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(x, y), sigT(0, 0), albedo);
	//}
	//else {
	//	med = new HeterogeneousMedium<2>(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(x, y), sigT, albedo);
	//}
	multiHeterogeneousMedium<2> *med;
	med = new multiHeterogeneousMedium<2>(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(x, y), sigT, albedoList, block);

	// core computing
#pragma omp parallel for //schedule(dynamic, 1)
	for (int sample = 1; sample <= N_Sample; ++sample) {
		int tid = omp_get_thread_num();
		Sampler &sampler = samplers[tid];

		const int maxDepth = 1000;

		// Initial postion and direction
		Eigen::Vector2d pos(sampler.nextSample()*x, y);
		Eigen::Vector2d dir(0.0, -1.0);

		double weight = 1.0;
		double refl = 0.0;
		double newWeight = 0.0;
		int intersectID = 0;;

		for (int dep = 1; dep <= maxDepth; ++dep) {
			// sample distance
			double dist;
			if (med->sampleDistance(pos, dir, sampler, dist)) {
				// update new position and direction
				pos = pos + dir * dist;
				dir = med->sampleDirection(pos, dir, sampler);

				if (dep <= 10)
					weight *= med->getAlbedo(pos, dir, sampler);
				else
					if (sampler.nextSample() > med->getAlbedo(pos, dir, sampler))
						break;
#ifdef nextEvent
				// sample a point "a" on receiptor
				Eigen::Vector2d a = med->sampleOnReceiptor(sampler, receiptorSize);
				double disToA = (a - pos).norm();
				double costheta = std::abs(pos(1) - a(1)) / disToA;
				double newWeight = med->evalAttenuation(pos, a, sampler) * (1.0 / (2.0 * PI)) * weight * receiptorSize * (costheta / disToA);
				refl += newWeight;
			}
			else {
				break;
			}
		}
#else
			}
			else {
				if (med->intersectReceiptor(pos, dir, sampler, receiptorSize, intersectID)) {
				//if (med->intersectReceiptor(pos, dir, sampler, receiptorSize)) {
					refl = weight;
				}
				break;
			}
		}
#endif
					
		// for each sample ray, we compute a refl and add them up
		if (intersectID != 0) reflectance(tid, intersectID-1) += refl;
		if (intersectID != 0) reflectance2(tid, intersectID-1) += refl * refl;
	}
	 
	reflectanceBlock = reflectance.colwise().sum();
	reflectanceBlock = reflectanceBlock / N_Sample;
	reflectanceTotal = reflectanceBlock.sum();

	reflectanceBlock2 = reflectance2.colwise().sum();
	reflectanceBlock2 = reflectanceBlock2 / N_Sample;
	reflectanceTotal2 = reflectanceBlock2.sum();

	for (i = 0; i < block; ++i)
		reflectanceStderrBlock[i] = sqrt(reflectanceBlock2[i] - reflectanceBlock[i] * reflectanceBlock[i]) / sqrt(N_Sample);
	
	reflectanceStderrTotal = sqrt(reflectanceTotal2 - reflectanceTotal * reflectanceTotal) / sqrt(N_Sample);

	std::ofstream outfile;

	outfile.open("tmp/reflectance.csv");
	outfile << std::fixed;
	outfile << std::setprecision(15) << reflectanceTotal;
	for (i = 0; i < block; ++i)
		outfile << "," << std::setprecision(15) << reflectanceBlock[i];
	outfile.close();

	outfile.open("tmp/reflectanceStderr.csv");
	outfile << std::fixed;
	outfile << std::setprecision(15) << reflectanceStderrTotal;
	for(i = 0; i < block; ++i)
		outfile << "," << std::setprecision(15) << reflectanceStderrBlock[i];
	outfile.close();

	return 0;
}
