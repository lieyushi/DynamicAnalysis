#pragma once
#include<cmath>
#include<cassert>
#include<iostream>
#include<cstring>

#ifndef PI
	#define PI 3.1415926535897
#endif

#ifndef UNIT
	#define UNIT 100
#endif

class ABC_flow
{
public:
	ABC_flow(void);
	~ABC_flow(void);
	static void generate_flow_position(float **data, const int& particle);
	static void generate_flow_velocity(float **data, const float& times, const int& particle, 
									   const float& time_step, const int& choice, const int& T);
	static float A;
	static float B;
	static float C;

private:
	static void generate_RK2_position(float *position, const float* velocity, const float& times, 
								      const float& time_step, const int& choice, const int& T);
	static void getRK2(float *position, float *velocity, const float& times, const float& time_step, const int& choice);
	static void getVelocity(float *position, float *velocity, const float& times, const float& time_step, const int& choice);
};

