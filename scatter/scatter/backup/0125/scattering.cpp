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
	const double albedo = std::atof(argv[2]);
	const int N_Sample = std::atoi(argv[3]);
	const int h_sigT = std::atoi(argv[4]);
	const int w_sigT = std::atoi(argv[5]);
	const double y = std::atof(argv[6]);
	const double x = std::atof(argv[7]);
	const double receiptorSize = std::atof(argv[8]);


	// read sigT from csv
	Eigen::MatrixXd sigT(h_sigT, w_sigT);
	for (int i = 0; i < h_sigT; ++i) {
		std::string line;
		std::getline(file, line);

		std::stringstream iss(line);
		for (int j = 0; j < w_sigT; ++j) {
			std::string val;
			std::getline(iss, val, ',');
			std::stringstream convertor(val);
			convertor >> sigT(i,j);
		}
	}

	// define output reflectance
	double reflectanceTotal = 0.0;
	Eigen::VectorXd reflectance = Eigen::VectorXd::Zero(nworkers);

	double reflectanceTotal2 = 0.0;
	Eigen::VectorXd reflectance2 = Eigen::VectorXd::Zero(nworkers);

	double reflectanceStderr = 0.0;

	// define medium 
	Medium<2> *med;
	if (sigT.maxCoeff() - sigT.minCoeff() < 0.0000001) {
		med = new HomogeneousMedium<2>(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(x, y), sigT(0, 0), albedo);
	}
	else {
		med = new HeterogeneousMedium<2>(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(x, y), sigT, albedo);
	}

	// core computring
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

		for (int dep = 1; dep <= maxDepth; ++dep) {
			// sample distance
			double dist;
			if (med->sampleDistance(pos, dir, sampler, dist)) {
				if (dep <= 10)
					weight *= albedo;
				else
					if (sampler.nextSample() > albedo)
						break;

				// update new position and direction
				pos = pos + dir * dist;
				dir = med->sampleDirection(pos, dir, sampler);

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
				if (med->intersectReceiptor(pos, dir, sampler, receiptorSize)) {
					refl += weight;
				}
				break;
			}
		}
#endif
					
		// for each sample ray, we compute a refl and add them up
		reflectance(tid) += refl;
		reflectance2(tid) += refl * refl;
	}
	 
	for (i = 0; i < nworkers; ++i)
		reflectanceTotal += reflectance(i);
	reflectanceTotal = reflectanceTotal / N_Sample;
	
	for (i = 0; i < nworkers; ++i)
		reflectanceTotal2 += reflectance2(i);
	reflectanceTotal2 = reflectanceTotal2 / N_Sample;

	reflectanceStderr = sqrt(reflectanceTotal2 - reflectanceTotal * reflectanceTotal) / sqrt(N_Sample);

	std::ofstream outfile;

	outfile.open("output/reflectance.csv");
	outfile << std::fixed;
	outfile << std::setprecision(15) << reflectanceTotal;
	outfile.close();

	outfile.open("output/reflectanceStderr.csv");
	outfile << std::fixed;
	outfile << std::setprecision(15) << reflectanceStderr;
	outfile.close();

	return 0;
}
