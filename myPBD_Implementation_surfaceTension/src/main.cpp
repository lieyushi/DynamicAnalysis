#include<iostream>
#include<fstream>
#include<iostream>
#include<fstream>
#include<sstream>
#include<climits>
#include<sys/time.h>
#include<cassert>

#include "SPH.h"

using namespace std;

const float factor_epsilon = 0.3;
const float vorticity_epsilon = 5.0;
SPH *sph = NULL;
clock_t starting, ending;
const int STEP = 2;
int START;
int artificialPressure, vorticityConfinement, surfaceTension;

float extrema[6] = {FLT_MAX, FLT_MIN, FLT_MAX, FLT_MIN, FLT_MAX, FLT_MIN};

class myTimer
{
public:
	double neighSearchTime, solvePressureTime, velocityUpdateTime, fileWrittenTime;
	myTimer(): neighSearchTime(0), solvePressureTime(0), velocityUpdateTime(0), fileWrittenTime(0)
	{	
	} 
};


void initPBF();
void initPBF(string& file_);
void runPBF();
void runPBF(const int& frame, myTimer& recorder, 
			const int& artificialPressure, 
			const int& vorticityConfinement);
void addTimerInfo(const myTimer& recorder, 
				  std::vector<string>& timeStamp, 
				  std::vector<double>& timeDiff);
void readme(const std::vector<string>& timeStamp, 
			const std::vector<double>& timeDiff);

int main(int argc, char* argv[])
{
	std::cout << "Please choose whether to use artificial pressure? 1.Yes, 0.No" << std::endl;
	std::cin >> artificialPressure;
	assert(artificialPressure==1 || artificialPressure==0);

	std::cout << "Please choose whether to use vorticity confinement? 1.Yes, 0.No" << std::endl;
	std::cin >> vorticityConfinement;
	assert(vorticityConfinement==1 || vorticityConfinement==0);

	std::cout << "Please choose surface tension type?\n" 
			  << "1.WCSPH, 2.versatile, 3.He2014, others:no" << std::endl;
	std::cin >> surfaceTension;
	
	struct timeval start, end;
	double difference;
	myTimer recorder;

	std::vector<string> timeStamp;
	std::vector<double> timeDiff;

	gettimeofday(&start, NULL);
	const int& frame = 1500;	
	//string file_ = "../sourceData/Frame 221.txt";
	//initPBF(file_);
	initPBF();
	gettimeofday(&end, NULL);
	difference = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
	timeStamp.push_back("Sampling and initializing pbf time: ");
	timeDiff.push_back(difference);

	for (int i = START + 1; i < START + frame; i++)
	{
		runPBF(i,recorder, artificialPressure, vorticityConfinement);
	}
	gettimeofday(&end, NULL);
	difference = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
	stringstream ss;
	ss << frame << " simulation step time: ";
	timeStamp.push_back(ss.str());
	timeDiff.push_back(difference);

	addTimerInfo(recorder, timeStamp, timeDiff);
	readme(timeStamp, timeDiff);
	if(sph)
	{
		delete sph;
		sph = NULL;
	}
	timeStamp.clear();
	timeDiff.clear();
	return 0;
}

void initPBF()
{	
	const float& interval = 1.0;
	const float& time_step = 0.016;
	const int& iteration = 4;
	const int even[3] = {60, 50, 65};
	START = 0;
	sph = new SPH( even, time_step, iteration, interval);
	sph->compute_boundary_neighbor();
	sph->computePsi();
	sph->printBoundaryParticlesVTK();
}

void initPBF(string& file_)
{
	const int& blank = file_.find(" ");
	file_.erase(0, blank+1);
	START = atoi(file_.c_str());
	cout << (START++) << endl;
	sph = new SPH(("../sourceData/Frame " + file_).c_str());
	sph->compute_boundary_neighbor();
	sph->computePsi();
}



void runPBF(const int& frame, myTimer& recorder, 
			const int& artificialPressure, 
			const int& vorticityConfinement)
{	
	cout << "Frame " << frame << ": " << endl;
	struct timeval beginning, ending;
	for (int i = 0; i < STEP; i++)
	{
		sph->update_position_by_force();
		std::cout << "force udpate!" << std::endl;

		gettimeofday(&beginning, NULL);
		sph->configure();
		sph->compute_neighbor();
		gettimeofday(&ending, NULL);
		recorder.neighSearchTime += ((ending.tv_sec  - beginning.tv_sec) * 1000000u 
								 + ending.tv_usec - beginning.tv_usec) / 1.e6;	

		gettimeofday(&beginning, NULL);					
		sph->update_position_by_iteration(factor_epsilon, artificialPressure); 
		std::cout << "Pressure solution!" << std::endl;
		gettimeofday(&ending, NULL);
		recorder.solvePressureTime += ((ending.tv_sec  - beginning.tv_sec) * 1000000u 
								   + ending.tv_usec - beginning.tv_usec) / 1.e6;

		gettimeofday(&beginning, NULL);
		sph->update_position_by_vorticity_and_XSPH(vorticity_epsilon, vorticityConfinement,
												   surfaceTension);
		std::cout << "Velocity update!" << std::endl;
		gettimeofday(&ending, NULL);
		recorder.velocityUpdateTime += ((ending.tv_sec  - beginning.tv_sec) * 1000000u 
									+ ending.tv_usec - beginning.tv_usec) / 1.e6;
	}
	gettimeofday(&beginning, NULL);
	sph->output(frame, extrema);
	gettimeofday(&ending, NULL);
	recorder.fileWrittenTime += ((ending.tv_sec  - beginning.tv_sec) * 1000000u
							 + ending.tv_usec - beginning.tv_usec) / 1.e6;
	cout << endl;
}

void readme(const std::vector<string>& timeStamp, const std::vector<double>& timeDiff)
{
	ofstream readme("README", ios::out);
	if(readme.fail())
	{
		cout << "Error creating files!" << endl;
		exit(-1);
	}
	readme << "-----------------------Simulation Option------------------------" << std::endl;
	readme << "		Artificial pressure: " << artificialPressure << std::endl;
	readme << "     Vorticity confinement: " << vorticityConfinement << std::endl;
	readme << std::endl;
	readme << "-----------------------Simulation Domain------------------------" << std::endl;
	readme << "		Boundary condition is: [" << sph->boundary.x_range.x << ", " << sph->boundary.x_range.y
	       << "] X [" << sph->boundary.y_range.x << ", " << sph->boundary.y_range.y
	       << "] X [" << sph->boundary.z_range.x << ", " << sph->boundary.z_range.y << "]!" << endl;
	readme << std::endl;
	readme << "-----------------------Simulation Scalar Extrema------------------------" << std::endl;
	readme << "		Velocity is: [" << extrema[0] << ", " << extrema[1] << "]!" << endl; 
	readme << "		Density is: [" << extrema[2] << ", " << extrema[3] << "]!" << endl; 
	readme << "		Vorticity is: [" << extrema[4] << ", " << extrema[5] << "]!" << endl; 
	readme << std::endl;
	readme << "-----------------------Simulation Timer Statistics------------------------" << std::endl;
	for (int i = 0; i < timeStamp.size(); ++i)
	{
		readme << "		" << timeStamp[i] << timeDiff[i] << " seconds." << std::endl;
	}
	readme << std::endl;
	readme << "-----------------------Simulation End------------------------" << std::endl;
	readme.close();
}


void addTimerInfo(const myTimer& recorder, std::vector<string>& timeStamp, std::vector<double>& timeDiff)
{
	timeStamp.push_back("Neighboring search time: ");
	timeDiff.push_back(recorder.neighSearchTime);
	timeStamp.push_back("Solving pressure iteration time: ");
	timeDiff.push_back(recorder.solvePressureTime);
	timeStamp.push_back("Velocity update time: ");
	timeDiff.push_back(recorder.velocityUpdateTime);
	timeStamp.push_back("FILE writing time: ");
	timeDiff.push_back(recorder.fileWrittenTime);
}