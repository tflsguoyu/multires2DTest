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

void readCSV(string filename, double data[][320]) {

	std::ifstream file(filename);

	for (int row = 0; row < 320; row++)
	{
		std::string line;
		std::getline(file, line);
		if (!file.good())
			break;

		std::stringstream iss(line);

		for (int col = 0; col < 320; col++)
		{
			std::string val;
			std::getline(iss, val, ',');
			std::stringstream convertor(val);
			convertor >> data[row][col];
		}
	}

}

void getCoord(double x, double y, int H, int W, int &r, int &c) {
	
	r = ceil((1.0 - y)*H);
	c = ceil(x * W);
	r = r == 0 ? 1 : r; c = c == 0 ? 1 : c;
	r = r - 1; c = c - 1;
}

int main(int argc, char *argv[]) {
    //int nworkers = omp_get_num_procs();
    //omp_set_num_threads(nworkers);
	//#pragma omp parallel for schedule(dynamic, 1)
	//#pragma omp critical
	rng.init(1);

	const int h_sigmaT_d = 320;
	const int w_sigmaT_d = 320;

	double sigmaT_d_NN[h_sigmaT_d][w_sigmaT_d];
	readCSV(argv[1], sigmaT_d_NN);
		
	const double albedo = atof(argv[2]);
	const int N_Sample = atoi(argv[3]);

	const int mapSize = 32;
	double reflectance = 0.0;
	double densityMap[mapSize][mapSize] = { { 0 } };

	for (int i = 1; i < N_Sample; i++) {

		const int maxDepth = 1000;
		double x[2] = {rng(),1.0};
		double w[2] = { 0.0,1.0 };

		int r, c;
		getCoord(x[0], x[1], h_sigmaT_d, w_sigmaT_d, r, c);

		double weight = 1.0 / N_Sample;

		for (int dep = 1; dep < maxDepth; dep++) {

			double t = -log(rng()) / sigmaT_d_NN[r][c];
			x[0] = x[0] - t * w[0]; x[1] = x[1] - t * w[1];

			if (x[1] > 1.0) {
				double intersectP_x = x[0] + (1 - x[1]) * w[0] / w[1];
				if (intersectP_x > 0 && intersectP_x < 1) {
					//reflectance += weight / sigmaT_d_NN[r][c];
					reflectance += weight;
					break;
				}
				else
					break;
			}
			else if (x[0] < 0.0 || x[0] > 1.0 || x[1] < 0.0)
				break;
			
			double theta = 2.0 * PI * rng();
			w[0] = cos(theta); w[1] = sin(theta);
			weight *= albedo;

			getCoord(x[0], x[1], h_sigmaT_d, w_sigmaT_d, r, c);

			int row, col;
			getCoord(x[0], x[1], mapSize, mapSize, row, col);

			densityMap[row][col] += weight / sigmaT_d_NN[r][c];

		}
	}

	ofstream outfile;
	outfile.open("output/densityMap.csv");
	for (int i = 0; i<mapSize; i++)
	{
		outfile << densityMap[i][0];

		for (int j = 1; j<mapSize; j++)
		{
			outfile << "," << densityMap[i][j];
		}

		outfile << endl;
	}
	outfile.close();

	outfile.open("output/reflectance.csv");
	outfile << reflectance;
	outfile.close();

	return 0;
}
