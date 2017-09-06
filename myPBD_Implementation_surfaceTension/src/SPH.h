#pragma once
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <glm/gtx/intersect.hpp> 
#include "Particle.h"
#include "Block.h"
#include "GL/glut.h"
#include "SurfaceTension.h"

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

	SPH(const int even[3], const float& time_step, const int& iteration, const float& interval);
	SPH(const string& file_); 
	void update_position_by_force();
	void configure();
	void compute_neighbor();
	void update_position_by_iteration( const float& factor_epsilon, const int& artificialPressure);
	void update_position_by_vorticity_and_XSPH(const float& vorticity_epsilon, const int& vorticityConfinement,
											   const int& surfaceOption);
	void output(const int& frame, float extrema[6]);
    void rendering();
	void compute_boundary_neighbor();
	void computePsi();
	void printBoundaryParticlesVTK();

private:
	const std::vector<int> find_block_by_layer(const int index[3], const int *layer);
	void addWall(const float *min_, const float *max_, const float& interval, const float& radius, float *normal);
	void addBunny(const char* fileName, const float& radius);
	void addObstacle(const float& interval, const float& radius);
	void addObstacle();
	void addBoundaryParticles(const float& interval, const float& radius);
	void addWaterVolume(const float& mass, const float& supportRadius, 
						 const float external_force[], const float& interval, const int even[]);
	void addBunnyWater(const string& fileName, const float& mass, const float& supportRadius, const float& interval,
						const float external_force[]);
	void printBoundary();
	void printObstacle(const Grid& cube, const int& order);
	void getVertexFace(const string& fileName, std::vector<std::vector<float> >& vertices, 
					   std::vector<std::vector<int> >& triangles, float limit[3][2]);
	bool stayInVolume(const float& x, const float& y, const float& z, 
					  const std::vector<std::vector<float> >& vertices, 
					  const std::vector<std::vector<int> >& triangles) const;
};



 