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

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
//using namespace cv;
#define PI 3.1415926535897932384626433832795

/*
 * Thread-safe random number generator
 */

struct RNG {
    RNG() : distrb(0.0, 1.0), engines() {}

    void init(int nworkers) {
        std::random_device rd;
        engines.resize(nworkers);
        for ( int i = 0; i < nworkers; ++i )
            engines[i].seed(rd());
    }

    double operator()() {
        int id = omp_get_thread_num();
        return distrb(engines[id]);
    }

    std::uniform_real_distribution<double> distrb;
    std::vector<std::mt19937> engines;
} rng;

//void readCSV(string filename, double data[][res]) {
//
//	std::ifstream file(filename);
//
//	for (int row = 0; row < res; row++)
//	{
//		std::string line;
//		std::getline(file, line);
//		if (!file.good())
//			break;
//
//		std::stringstream iss(line);
//
//		for (int col = 0; col < res; col++)
//		{
//			std::string val;
//			std::getline(iss, val, ',');
//			std::stringstream convertor(val);
//			convertor >> data[row][col];
//		}
//	}
//
//}

void getCoord(double x, double y, int H, int W, int &r, int &c) {
	
	r = ceil((1.0 - y)*H);
	c = ceil(x * W);
	r = r == 0 ? 1 : r; c = c == 0 ? 1 : c;
	r = r - 1; c = c - 1;
}

int main(int argc, char *argv[]) {
    int nworkers = omp_get_num_procs();
    omp_set_num_threads(nworkers);
	rng.init(nworkers);	
	
	const double albedo = atof(argv[2]);
	const int N_Sample = atoi(argv[3]);
	const int h_sigmaT_d = atoi(argv[4]);
	const int w_sigmaT_d = atoi(argv[5]);
	const double h = atof(argv[6]);
	const double w = atof(argv[7]);

	// read csv
	double **sigmaT_d_NN = new double*[h_sigmaT_d];
	for (int i = 0; i < h_sigmaT_d; i++)
		sigmaT_d_NN[i] = new double[w_sigmaT_d];
	
	ifstream file(argv[1]);
	for (int row = 0; row < h_sigmaT_d; row++)
	{
		string line;
		getline(file, line);
		if (!file.good())
			break;

		stringstream iss(line);
		for (int col = 0; col < w_sigmaT_d; col++)
		{
			string val;
			getline(iss, val, ',');
			stringstream convertor(val);
			convertor >> sigmaT_d_NN[row][col];
		}
	}

	// compute MAX of sigmaT
	double sigmaT_MAX = 0.0;
	for (int i = 0; i < h_sigmaT_d; i++)
		for (int j = 0; j < w_sigmaT_d; j++) {
			if (sigmaT_d_NN[i][j] > sigmaT_MAX)
				sigmaT_MAX = sigmaT_d_NN[i][j];
		}

	// create density map 
	//int h_mapSize = 32;
	//int w_mapSize = h_mapSize * round(w / h);
	//double **densityMap = new double*[h_mapSize];
	//for (int i = 0; i < h_mapSize; i++)
	//	densityMap[i] = new double[w_mapSize];
	//for (int i = 0; i < h_mapSize; i++)
	//	for (int j = 0; j < w_mapSize; j++)
	//		densityMap[i][j] = 0;

	double reflectanceTotal = 0.0;
	double *reflectance = new double[nworkers];
	for (int i = 0; i < nworkers; i++) reflectance[i] = 0;

#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 1; i < N_Sample; i++) {

		int tid = omp_get_thread_num();

		const int maxDepth = 1000;
		double x[2] = {rng()*w, h};
		double d[2] = { 0.0, 1.0 };

		int r, c;
		int row, col;
		double sigmaT, sigmaT_next;
		double x_next[2];
		int r_next, c_next;

		double weight = 1.0 / N_Sample;

		for (int dep = 1; dep < maxDepth; dep++) {

			// Method 1: 
			//double t = -log(rng()) / sigmaT;

			// Method 2: Woodcock
			double t = 0.0;
			while (1) {
				t = t - log(rng()) / sigmaT_MAX;
				x_next[0] = x[0] - t * d[0]; x_next[1] = x[1] - t * d[1];
				if (x_next[0] < 0 || x_next[0]>w || x_next[1] < 0 || x_next[1]>h)
					break;
				getCoord(x_next[0]/w, x_next[1]/w, h_sigmaT_d, w_sigmaT_d, r_next, c_next);
				sigmaT_next = sigmaT_d_NN[r_next][c_next];
				if ((sigmaT_next / sigmaT_MAX) > rng())
					break;
			}
			
			x[0] = x[0] - t * d[0]; x[1] = x[1] - t * d[1];

			if (x[1] > h) {
				double intersectP_x = x[0] + (h - x[1]) * d[0] / d[1];
				if (intersectP_x > 0 && intersectP_x < w) {
					reflectance[tid] += weight;
					break;
				}
				else
					break;
			}
			else if (x[0] < 0.0 || x[0] > w || x[1] < 0.0)
				break;
			
			double theta = 2.0 * PI * rng();
			d[0] = cos(theta); d[1] = sin(theta);
			
			getCoord(x[0]/w, x[1]/h, h_sigmaT_d, w_sigmaT_d, r, c);
			//getCoord(x[0]/w, x[1]/h, h_mapSize, w_mapSize, row, col);

			sigmaT = sigmaT_d_NN[r][c];
			//densityMap[row][col] += weight / sigmaT;

			weight *= albedo;

		}
	}
//#pragma omp critical
	for (int i = 0; i < nworkers; i++)
		reflectanceTotal += reflectance[i];


	ofstream outfile;
	//outfile.open("output/densityMap.csv");
	//for (int i = 0; i<h_mapSize; i++)
	//{
	//	outfile << densityMap[i][0];

	//	for (int j = 1; j<w_mapSize; j++)
	//	{
	//		outfile << "," << densityMap[i][j];
	//	}

	//	outfile << endl;
	//}
	//outfile.close();

	outfile.open("output/reflectance.csv");
	outfile << reflectanceTotal;
	outfile.close();

	//for (int i = 0; i < h_mapSize; i++)
	//	delete[] densityMap[i];
	//delete[] densityMap;

	for (int i = 0; i < h_sigmaT_d; i++)
		delete[] sigmaT_d_NN[i];
	delete[] sigmaT_d_NN;

	delete[] reflectance;

	return 0;
}
