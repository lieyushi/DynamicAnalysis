#include "ABC_flow.h"
#include <omp.h>
float ABC_flow::A = sqrt(3.0);
float ABC_flow::B = sqrt(2.0);
float ABC_flow::C = 1.0;

const int& N = UNIT+1;

ABC_flow::ABC_flow(void)
{
}


ABC_flow::~ABC_flow(void)
{
}


void ABC_flow::generate_flow_position(float **data, const int& particle)
{
	std::cout << "ABC flow data generation starts!" << std::endl;
	const float& step = 2.0/(float)UNIT*PI;

#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particle; ++i)
	{
		const int& a = i%N;
		const int& b = i/N%N;
		const int& c = i/N/N;
		data[i][0] = a*step;
		data[i][1] = b*step;
		data[i][2] = c*step;
		data[i][46] = a*step;
		data[i][47] = b*step;
		data[i][48] = c*step;
	}
}


void ABC_flow::generate_flow_velocity(float **data, const float& times, const int& particle, const float& time_step, 
									  const int& choice, const int& T)
{
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < particle; ++i)
	{
		getVelocity(&data[i][0], &data[i][3], times, time_step, choice);
		generate_RK2_position(&data[i][46], &data[i][3], times, time_step, choice, T);
	}
}


void ABC_flow::generate_RK2_position(float *position, const float* velocity, const float& times, const float& time_step, 
									 const int& choice, const int& T)
{
	float tempVelo[3];
	memcpy(tempVelo, velocity, 3*sizeof(float));
	for (int j = 0; j < T; ++j)
	{
		getRK2(position, tempVelo, times+j, time_step, choice);
		getVelocity(position, tempVelo, times+j+1, time_step, choice);
	}
}

/************************* Use RK-2 interpolation to predict future position ****************************************/
void ABC_flow::getRK2(float *position, float *velocity, const float& times, const float& time_step, const int& choice)
{
	float tempPosi[3];
	for (int i = 0; i < 3; ++i)
	{
		tempPosi[i] = position[i]+velocity[i]*0.5*time_step;
	}
	getVelocity(tempPosi, velocity, times+0.5, time_step, choice);
	for (int i = 0; i < 3; ++i)
	{
		position[i] += velocity[i]*time_step;
	}
}


void ABC_flow::getVelocity(float *position, float *velocity, const float& times, const float& time_step, const int& choice)
{
	const float& t = times*time_step;
	float coefficient;

	switch(choice)
	{
	default:
	case 0:
	 	{	 		
	 		coefficient = (A+(1-exp(-0.1*t))*(sin(2*PI*t)));
 			velocity[0] = coefficient*sin(position[2]) + C*cos(position[1]);
			velocity[1] = B*sin(position[0]) + coefficient*cos(position[2]);
			velocity[2] = C*sin(position[1]) + B*cos(position[0]); 
	 	}
	 	break;


	case 1:
		{	 		
	 		coefficient = (A + 0.5*t*(sin(PI*t)));
 			velocity[0] = coefficient*sin(position[2]) + B*cos(position[1]);
			velocity[1] = B*sin(position[0]) + C*cos(position[2]);
			velocity[2] = C*sin(position[1]) + coefficient*cos(position[0]); 
		}
		break;


	case 2:
		{ 		
	 		coefficient = (A+0.5*t*(sin(PI*t)));
 			velocity[0] = coefficient*sin(position[2]) + C*cos(position[1]);
			velocity[1] = B*sin(position[0]) + coefficient*cos(position[2]);
			velocity[2] = C*sin(position[1]) + B*cos(position[0]); 
		}
		break;


	case 3:
		{				
			velocity[0] = A*sin(position[2]) + C*cos(position[1]);
			velocity[1] = B*sin(position[0]) + A*cos(position[2]);
			velocity[2] = C*sin(position[1]) + B*cos(position[0]);
		}
		break;

	}
}
