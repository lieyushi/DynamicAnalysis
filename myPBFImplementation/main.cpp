#include<iostream>
#include<fstream>
#include "SPH.h"
#include<climits>

using namespace std;

const float factor_epsilon = 1.0e-6;
const float vorticity_epsilon = 0.01;
SPH *sph = NULL;
clock_t start, end;
const int STEP = 1;
int START;

float extrema[6] = {INT_MAX, INT_MIN, INT_MAX, INT_MIN, INT_MAX, INT_MIN};
//velocity, density, vorticity


void initPBF();
void initPBF(string& file_);
void runPBF();
void runPBF(const int& frame);
void readme();


int main(int argc, char* argv[])
{

	time(&start);
	const int& frame = 1500;
	
	//string file_ = "Frame 1099.txt";
	//initPBF(file_);

	initPBF();
	for (int i = START + 1; i < START + frame; i++)
	{
		runPBF(i);
	}
	delete sph;
	time(&end);
	readme();
	return 0;
}


void initPBF()
{	
	const int& number = 128000;
	const float& interval = 0.05;
	const float& time_step = 0.004;
	const int& iteration = 3;
	const int even[3] = {40, 40, 80};
	START = 0;
	sph = new SPH( even, number, time_step, iteration, interval);
	sph->compute_boundary_neighbor();
	sph->computePsi();
}

void initPBF(string& file_)
{
	file_.erase(0, 6);
	START = atoi(file_.c_str());
	cout << START << endl;
	sph = new SPH(("Frame " + file_).c_str());
	sph->compute_boundary_neighbor();
	sph->computePsi();
}



void runPBF(const int& frame)
{	
	cout << "Frame " << frame << ": " << endl;
	for (int i = 0; i < STEP; i++)
	{
		sph->update_position_by_force();
		sph->configure(frame, extrema);
		sph->compute_neighbor();		
		sph->update_position_by_iteration(factor_epsilon); 
		sph->update_position_by_vorticity_and_XSPH(vorticity_epsilon);
	}
	sph->output(frame, extrema);
	cout << endl;
}


void readme()
{
	ofstream readme("README", ios::out);
	if(readme.fail())
	{
		cout << "Error creating files!" << endl;
		exit(-1);
	}
	readme << "Boundary condition is: [" << sph->boundary.x_range.x << ", " << sph->boundary.x_range.y
	       << "] X [" << sph->boundary.y_range.x << ", " << sph->boundary.y_range.y
	       << "] X [" << sph->boundary.z_range.x << ", " << sph->boundary.z_range.y << "]!" << endl;
	readme << "Time is: " << float(difftime(end,start)) << " S!" << endl;
	readme << "Velocity is: [" << extrema[0] << ", " << extrema[1] << "]!" << endl; 
	readme << "Density is: [" << extrema[2] << ", " << extrema[3] << "]!" << endl; 
	readme << "Vorticity is: [" << extrema[4] << ", " << extrema[5] << "]!" << endl; \
	readme.close();
}