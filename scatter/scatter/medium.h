#pragma once
#include <Eigen/Dense>
#include "sampler.h"

#define PI 3.1415926535897932384626433832795

template <int ndim>
class Medium
{
public:
	typedef Eigen::Matrix<double, ndim, 1> VectorType;

	virtual bool sampleDistance(const VectorType &p, const VectorType &d,
		Sampler &sampler, double &dist) const = 0;

	virtual VectorType sampleDirection(const VectorType &p, const VectorType &d,
		Sampler &sampler) const = 0;

	virtual double getAlbedo(const VectorType &p, const VectorType &d,
		Sampler &sampler) const = 0;

	virtual bool intersectReceiptor(const VectorType &p, const VectorType &d,
		Sampler &sampler, const double &receiptorWidth) const = 0;

	virtual VectorType sampleOnReceiptor(
		Sampler &sampler, const double &receiptorWidth) const = 0;

	virtual double evalAttenuation(const VectorType &p1, const VectorType &p2,
		Sampler &sampler) const = 0;
};


template <int ndim>
class HomogeneousMedium : public Medium<ndim>
{
public:
	//typedef typename Medium<ndim>::VectorType VectorType;
	typedef Medium<ndim> Parent;
	using typename Parent::VectorType;

	HomogeneousMedium(const VectorType &_pMin, const VectorType &_pMax, const double &_sigT, const double &_albedo)
		: pMin(_pMin), pMax(_pMax), sigT(_sigT), albedo(_albedo) {}

	bool sampleDistance(const VectorType &p, const VectorType &d, Sampler &sampler, double &dist) const
	{
		dist = -std::log(sampler.nextSample()) / sigT;
		VectorType p1 = p + d * dist;
		if (p1[0] < pMin[0] || p1[0] > pMax[0] || p1[1] < pMin[1] || p1[1] > pMax[1]) {
			//
			return false;
		}
		else {
			return true;
		}
	}

	VectorType sampleDirection(const VectorType &p, const VectorType &d, Sampler &sampler) const
	{
		double theta = 2.0 * PI * sampler.nextSample();
		return VectorType(std::cos(theta), std::sin(theta));
	}

	double getAlbedo(const VectorType &p, const VectorType &d, Sampler &sampler) const
	{
		return albedo;
	}

	bool intersectReceiptor(const VectorType &p, const VectorType &d, Sampler &sampler, const double &receiptorWidth) const
	{
		if (d[1] > 0) {
			double intersectP_x = p[0] + (pMax[1] - p[1]) * d[0] / d[1];
			if (intersectP_x > (pMax[0] - receiptorWidth) / 2 && intersectP_x < (pMax[0] + receiptorWidth) / 2) {
				return true;
			}
		}
		return false;
	}

	VectorType sampleOnReceiptor(Sampler &sampler, const double &receiptorWidth) const
	{
		return VectorType(sampler.nextSample() * receiptorWidth + (pMax[0] - receiptorWidth) / 2, pMax[1]);
	}

	double evalAttenuation(const VectorType &p1, const VectorType &p2, Sampler &sampler) const
	{
		return std::exp(-sigT * (p1 - p2).norm());
	}


protected:
	VectorType pMin, pMax;
	double sigT, albedo;
};


template <int ndim>
class HeterogeneousMedium : public Medium<ndim>
{
public:
    //typedef typename Medium<ndim>::VectorType VectorType;
	typedef Medium<ndim> Parent;
	using typename Parent::VectorType;

	HeterogeneousMedium(const VectorType &_pMin, const VectorType &_pMax, const Eigen::MatrixXd &_sigT, const double &_albedo)
		: pMin(_pMin), pMax(_pMax), sigT(_sigT), sigT_MAX(_sigT.maxCoeff()), albedo(_albedo) {}

	bool sampleDistance(const VectorType &p, const VectorType &d, Sampler &sampler, double &dist) const
	{
		dist = 0.0;
		VectorType p1;
		while (1) {
			dist = dist - log(sampler.nextSample()) / sigT_MAX;
			p1 = p + d * dist;
			if (p1[0] < pMin[0] || p1[0] > pMax[0] || p1[1] < pMin[1] || p1[1] > pMax[1]) {
				return false;
			}
			if ((getSigT(p1) / sigT_MAX) > sampler.nextSample()) {
				return true;
			}
		}
	}

	VectorType sampleDirection(const VectorType &p, const VectorType &d, Sampler &sampler) const
	{
		double theta = 2.0 * PI * sampler.nextSample();
		return VectorType(std::cos(theta), std::sin(theta));
	}

	double getAlbedo(const VectorType &p, const VectorType &d, Sampler &sampler) const
	{
		return albedo;
	}

	bool intersectReceiptor(const VectorType &p, const VectorType &d, Sampler &sampler, const double &receiptorWidth) const
	{
		if (d[1] > 0) {
			double intersectP_x = p[0] + (pMax[1] - p[1]) * d[0] / d[1];
			if (intersectP_x > (pMax[0] - receiptorWidth) / 2 && intersectP_x < (pMax[0] + receiptorWidth) / 2) {
				return true;
			}
		}
		return false;
	}

	VectorType sampleOnReceiptor(Sampler &sampler, const double &receiptorWidth) const
	{
		return VectorType(sampler.nextSample() * receiptorWidth + (pMax[0] - receiptorWidth) / 2, pMax[1]);
	}

	double evalAttenuation(const VectorType &p1, const VectorType &p2, Sampler &sampler) const
	{
		int woodCockIteration = 1;
		double attn = 0.0;
		for (int woodcock_it = 0; woodcock_it < woodCockIteration; ++woodcock_it) {
			double dist;
			if (!sampleDistance(p1, (p2 - p1).normalized(), sampler, dist)) {
				attn += 1.0;
			}
		}
		return attn / static_cast<double>(woodCockIteration);
	}

protected:
	double getSigT(const VectorType &p) const {
		
		int r = static_cast<int>(std::ceil((1.0 - p[1] / pMax[1]) * sigT.rows()));
		int c = static_cast<int>(std::ceil(p[0] / pMax[0] * sigT.cols()));
		
		if (r < 0) r = 0;
		else if (r >= sigT.rows()) r = static_cast<int>(sigT.rows()) - 1;
		if (c < 0) c = 0;
		else if (c >= sigT.cols()) c = static_cast<int>(sigT.cols()) - 1;
		
		return sigT(r, c);
	}
	
	VectorType pMin, pMax;
	Eigen::MatrixXd sigT;
	double sigT_MAX;
	double albedo;
};


template <int ndim>
class multiHeterogeneousMedium : public HeterogeneousMedium<ndim>
{
public:
    //typedef typename HeterogeneousMedium<ndim>::VectorType VectorType;
	typedef HeterogeneousMedium<ndim> Parent;
	using typename Parent::VectorType;
	using Parent::pMin;
	using Parent::pMax;

	multiHeterogeneousMedium(const VectorType &_pMin, const VectorType &_pMax, const Eigen::MatrixXd &_sigT, const Eigen::VectorXd &_albedo, const int &_numOfBlock)
		: HeterogeneousMedium<ndim>(_pMin, _pMax, _sigT, 0.0), albedoList(_albedo), numOfBlock(_numOfBlock) {}

	bool intersectReceiptor(const VectorType &p, const VectorType &d, Sampler &sampler, const double &receiptorWidth, int &intersectID) const
	{
		if (d[1] > 0) {
			double intersectP_x = p[0] + (pMax[1] - p[1]) * d[0] / d[1];
			if (intersectP_x > (pMax[0] - receiptorWidth) / 2 && intersectP_x < (pMax[0] + receiptorWidth) / 2) {
				intersectID = static_cast<int>(std::ceil(intersectP_x / ((pMax[0] - pMin[0]) / numOfBlock)));
				return true;
			}
		}
		return false;
	}

	double getAlbedo(const VectorType &p, const VectorType &d, Sampler &sampler) const
	{
		int albedoID = static_cast<int>(std::ceil(p[0] / ((pMax[0] - pMin[0]) / numOfBlock)));
		return albedoList[albedoID-1];
	}

protected:
	int numOfBlock;
	Eigen::VectorXd albedoList;
};
