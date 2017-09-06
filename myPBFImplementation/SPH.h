#pragma once
#include "Particle.h"
#include<stdlib.h>
#include<time.h>
#include "Block.h"
#include<cmath>
#include <iomanip>
#include "GL/glut.h"
#include <cstring>

using namespace std;

class SPH
{
public:
	SPH(void);
	~SPH(void);
	std::vector<Particle > particle_list;
	std::vector<Particle > boundary_particle;
	int fluid_number;
	int boundary_number;
	Grid boundary;
	Grid obstacle;
	float time_step;
	std::vector<Block > block_storage;
	int block_number;
	int block_unit[3];
	float blockSize;
	int iteration;
	Grid obstackle;

	SPH(const int even[3], const int& number, const float& time_step, const int& iteration, const float& interval);
	SPH(const string& file_);
	//void display_boundary(); 
	void update_position_by_force();
	void configure();
	void compute_neighbor();
	void update_position_by_iteration( const float& factor_epsilon);
	void update_position_by_vorticity_and_XSPH(const float& vorticity_epsilon);
	void output(const int& frame, float extrema[6]);
    void rendering();
	void compute_boundary_neighbor();
	void computePsi();

private:
	const std::vector<int> find_block_by_layer(const int index[3], const int *layer);
	void addWall(const float *min_, const float *max_, const float& interval, const float& radius);
};



 