#pragma once
#include <random>

//class B
//{
//public:
//	B(int x) { value = x; }
//
//	int value;
//};
//
//class C
//{
//public:
//	C(int x, int y) { value1 = x; value2 = y; }
//
//	int value1, value2;
//};
//
//class A
//{
//public:
//	A(int bx, int cx, int cy)
//		: b(bx), c(cx, cy), value(100)
//	{
//	}
//
//	B b;
//	C c;
//	const int value;
//};


class Sampler
{
public:
	double nextSample() {
		return distrb(engine);
	}

    void init(int seed) {
		engine.seed(seed);
    }

protected:
    std::uniform_real_distribution<double> distrb;
    std::mt19937 engine;
};