#pragma once
#include <vector>
#include "Grid.h"
#include "Block.h"
#include <omp.h>
#include <iostream>
#include "Kernel.h"
#define gravity (float)9.81
const float rest_density = 1000;
const float ART_COEFFICIENT = 0.1;
const int ART_EXPONENTIAL = 4;
const float XSPHConst = 0.03;
const float DELTA = 0.1;

class Particle
{
public:
	Particle();

	Particle(const float posi[3], 
			 const float& radius, 
			 float *normal_);

	~Particle(void);

	Particle(const float& mass, 
			 const float posi[3], 
			 const float velo[3], 
			 const float& radius, 
			 const float force[3], 
			 const int& index);

	void compute_position_update_by_factor(const int& artificialPressure,
										   std::vector<Particle >& particle_vec, 
										   const std::vector<Particle>& boundary_particle );

	void compute_factor(const float& factor_epsilon, 
						std::vector<Particle >& particle_vec, 
						const std::vector<Particle>& boundary_particle);
	
	void compute_vorticity(std::vector<Particle >& particle_vec, 
						   const std::vector<Particle>& boundary_particle, 
						   const float& time_step);

	void get_velocity_by_vorticity(const float& vorticity_epsilon, 
								   std::vector<Particle >& particle_vec, 
								   const std::vector<Particle>& boundary_particle,
								   const float& time_step);

	void assigned_to_block( std::vector<Block>& block_list, 
							int *block_unit, 
							const float& blockSize, 
							const Grid& boundary);


	void collide_boundary_trivial(const Grid& grid);
	void boundaryClamp(const Grid& grid);


	float mass;
	std::vector<int > neighbor;
	glm::vec3 external_foce;
	glm::vec3 position;
	glm::vec3 original_position;
	glm::vec3 delta_position;
	glm::vec3 velocity;
	glm::vec3 normal;
	glm::vec3 vorticity;
	glm::vec3 acceleration;

	float pressure;
	float radius;
	float factor;
	int num_neigh;
	int index;
	float density;
	int posi_index[3];
};


const bool stay_in_grid(const glm::vec3& position, 
					    const Grid& grid, int tag[3]);

const int stay_in_interval(const float var, 
						   const glm::vec2& range);