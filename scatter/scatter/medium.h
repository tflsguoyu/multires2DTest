#pragma once
#include <Eigen/core>
#include "sampler.h"

#define PI 3.1415926535897932384626433832795

template <int ndim>
class Medium
{
public:
	typedef typename Eigen::Matrix<double, ndim, 1> VectorType;

	virtual bool sampleDistance(const VectorType &p, const VectorType &d,
		Sampler &sampler, double &dist) const = 0;

	virtual VectorType sampleDirection(const VectorType &p, const VectorType &d,
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
	typedef typename Medium<ndim> Parent;
	using Parent::VectorType;

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
	HeterogeneousMedium(const VectorType &_pMin, const VectorType &_pMax, const Eigen::MatrixXd &_sigT, const double &_albedo)
		: pMin(_pMin), pMax(_pMax), sigT(_sigT), sigT_MAX(_sigT.maxCoeff()), albedo(_albedo) {}

	bool sampleDistance(const VectorType &p, const VectorType &d, Sampler &sampler, double &dist) const
	{
		dist = 0.0;
		VectorType p1;
		while (1) {
			dist = dist - log(sampler.nextSample()) / sigT_MAX;
			p1 = p - d * dist;
			if (p1[0] < pMin[0] || p1[0] > pMax[0] || p1[1] < pMin[1] || p1[1] > pMax[1]) {
				return false;
			}
			if ((this->getSigT(p1) / sigT_MAX) > sampler.nextSample()) {
				return true;
			}
		}
	}

	VectorType sampleDirection(const VectorType &p, const VectorType &d, Sampler &sampler) const
	{
		double theta = 2.0 * PI * sampler.nextSample();
		return VectorType(std::cos(theta), std::sin(theta));
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
		int woodCockIteration = 10;
		double attn = 0.0;
		for (int woodcock_it = 0; woodcock_it < woodCockIteration; ++woodcock_it) {
			double dist;
			if (!this->sampleDistance(p1, p2 - p1, sampler, dist)) {
				attn += 1.0;
			}
		}
		return attn / static_cast<double>(woodCockIteration);
	}



protected:
	double getSigT(const VectorType &p) const {
		int r = std::ceil((1.0 - p[1] / pMax[1]) * sigT.rows());
		int c = std::ceil(p[0] / pMax[0] * sigT.cols());
		r = r == 0 ? 1 : r; c = c == 0 ? 1 : c;
		r = r - 1; c = c - 1;
		return sigT(r, c);
	}
	
	VectorType pMin, pMax;
	Eigen::MatrixXd sigT;
	double sigT_MAX;
	double albedo;
};
