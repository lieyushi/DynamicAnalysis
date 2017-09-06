#include "Particle.h"

#define ERROR_COLLISION 0.02

Particle::Particle(const float posi[3], 
				   const float& radius, 
				   float *normal_)
{
	mass = 0;
	position.x = posi[0]; position.y = posi[1]; position.z = posi[2]; 
	this->radius = radius;
	normal[0] = normal_[0], normal[1] = normal_[1], normal[2]=normal_[2];
}


Particle::~Particle(void)
{

}



Particle::Particle()
{

}



Particle::Particle(const float& mass, 
				   const float posi[3], 
				   const float velo[3], 
				   const float& radius, 
				   const float force[3], 
				   const int& index)
{
	this->mass = mass;
	velocity.x = velo[0]; velocity.y = velo[1]; velocity.z = velo[2];
	position.x = posi[0]; position.y = posi[1]; position.z = posi[2];
	original_position.x = posi[0]; original_position.y = posi[1]; original_position.z = posi[2];
	external_foce.x = force[0]; external_foce.y = force[1]; external_foce.z = force[2];
	pressure = (float)0;
	factor = (float)0;
	vorticity.x = 0; vorticity.y = 0; vorticity.z = 0;
	this->index = index;
	this->radius = radius;
	delta_position.x = delta_position.y = delta_position.z = (float)0;
	density = (float)0;
	num_neigh = 0;
	posi_index[0] = posi_index[1] = posi_index[2] = -1;
}




void Particle::assigned_to_block( std::vector<Block>& block_list, 
								  int *block_unit, 
								  const float& blockSize, 
								  const Grid& boundary)
{
	
	posi_index[0] = (int)floor((position.x - boundary.x_range.x)/blockSize);
	posi_index[1] = (int)floor((position.y - boundary.y_range.x)/blockSize);
	posi_index[2] = (int)floor((position.z - boundary.z_range.x)/blockSize); // calculating the index array of point based on grid

	assert(posi_index[0] >= 0 && posi_index[0] < block_unit[0]);
	assert(posi_index[1] >= 0 && posi_index[1] < block_unit[1]);
	assert(posi_index[2] >= 0 && posi_index[2] < block_unit[2]);

	/*if(!((posi_index[0] >= 0 && posi_index[0] < block_unit[0]) && (posi_index[1] >= 0 && posi_index[1] < block_unit[1])
	   && (posi_index[2] >= 0 && posi_index[2] < block_unit[2])))
	{
		std::cout << posi_index[0] << " " << posi_index[1] << " " << posi_index[2] << std::endl;
		std::cout << position.x << " " << position.y << " " << position.z << std::endl;
		std::cout << boundary.x_range.x << " " << boundary.y_range.x << " " << boundary.z_range.x << std::endl;
		std::cout << index << std::endl;
		exit(-1);
	}*/

	const int& block_index = posi_index[0] + block_unit[0] * posi_index[1] + block_unit[0] * block_unit[1] * posi_index[2];	
	assert(block_index >= 0 && block_index < block_list.size());
	block_list[block_index].add(index);	
}


void Particle::compute_factor(const float& factor_epsilon, 
							  std::vector<Particle >& particle_vec, 
							  const std::vector<Particle>& boundary_particle) 
{
	density = (float)0.0;
	if(num_neigh == 0)
		return;
	
	float sum_constraint = 0.0, unit;
	const int& vecSize = particle_vec.size();
	////for k = i case
	glm::vec3 temp, spiky_gradient;
	int neighIndex;
	Particle tempParticle;
	for(int i = 0; i < num_neigh; ++ i)
	{
		neighIndex=neighbor[i];
		if (neighIndex < vecSize)
		{
			tempParticle=particle_vec[neighIndex];
			spiky_gradient = -tempParticle.mass/rest_density*Kernel::W_spiky_gradient(position-tempParticle.position, radius);
			density += tempParticle.mass * Kernel::W_poly6(position-tempParticle.position, radius);
		}
		else
		{
			tempParticle=boundary_particle[neighIndex-vecSize];
			spiky_gradient = -tempParticle.mass/rest_density*Kernel::W_spiky_gradient(position-tempParticle.position, radius);
			density += tempParticle.mass * Kernel::W_poly6(position-tempParticle.position, radius);
		}
		temp -= spiky_gradient;  
		sum_constraint += glm::dot(spiky_gradient,spiky_gradient); 
	}
	sum_constraint += glm::dot(temp, temp);
	factor = - std::max(float(density/rest_density-1.0), float(0.0))/(sum_constraint + factor_epsilon);
}


void Particle::compute_position_update_by_factor( const int& artificialPressure,
												  std::vector<Particle >& particle_vec, 
												  const std::vector<Particle>& boundary_particle )
{
	delta_position = glm::vec3(0.0);
	if(num_neigh == 0)
		return;

	glm::vec3 grad;
	float upper, artificial, multi;
	Particle tempPar;

	const int& vecSize = particle_vec.size();

	const float delta_q = (float)DELTA * radius;
	const float& lower = Kernel::W_poly6(delta_q, radius);	
	for(int i = 0; i < num_neigh; ++ i)
	{
		if (neighbor[i]<vecSize)
		{
			tempPar=particle_vec[neighbor[i]];
			upper = Kernel::W_poly6(position-tempPar.position, radius);
			grad = tempPar.mass/rest_density*Kernel::W_spiky_gradient(position - tempPar.position, radius); 
			artificial = - (float)ART_COEFFICIENT * pow(upper / lower, ART_EXPONENTIAL);
			multi = factor + tempPar.factor;
			if(artificialPressure)
				multi += artificial;			
		}
		else
		{
			tempPar=boundary_particle[neighbor[i]-vecSize];
			upper = Kernel::W_poly6(position - tempPar.position, radius);
			artificial = - (float)ART_COEFFICIENT * pow(upper / lower, ART_EXPONENTIAL);
			grad = tempPar.mass/rest_density*Kernel::W_spiky_gradient(position - tempPar.position, radius); //The derivative should be under i
			multi = 2.0*factor;
			//multi = factor;
			if(artificialPressure)
				multi += artificial;		
		}
		delta_position += multi*grad;		
	}
}



void Particle::compute_vorticity( std::vector<Particle >& particle_vec, 
								  const std::vector<Particle>& boundary_particle, 
								  const float& time_step )
{
	if(num_neigh == 0)
		return;
	vorticity = glm::vec3(0.0);
	glm::vec3 spiky_gradient, change_XSPH, velocity_dist;
	Particle tempPar;
	const int& vecSize = particle_vec.size();
	float unit;
	for(int i = 0; i < num_neigh; i++)
	{
		if (neighbor[i] < vecSize)
		{
			tempPar=particle_vec[neighbor[i]];
			if(tempPar.density>1.0e-3)
				unit = tempPar.mass/tempPar.density;
			else
				unit = tempPar.mass/rest_density;
			spiky_gradient = -unit*Kernel::W_spiky_gradient(position - tempPar.position, radius);
			velocity_dist = tempPar.velocity - velocity;
			vorticity += glm::cross( velocity_dist, spiky_gradient);
			change_XSPH += unit*Kernel::W_poly6(position-tempPar.position, radius)*velocity_dist;
		}	
		else
		{
			tempPar = boundary_particle[neighbor[i]-vecSize];
			unit = tempPar.mass/rest_density;
			spiky_gradient = -unit*Kernel::W_spiky_gradient(position - tempPar.position, radius);
			velocity_dist = -glm::dot(velocity,tempPar.normal)*tempPar.normal;
			//velocity_dist = -velocity;
			vorticity += glm::cross( velocity_dist, spiky_gradient);
			change_XSPH += unit*Kernel::W_poly6(position - tempPar.position, radius) * velocity_dist;
		}
	}
	acceleration=XSPHConst*change_XSPH/time_step;
	//velocity+=XSPHConst*change_XSPH;
}


void Particle::get_velocity_by_vorticity(const float& vorticity_epsilon, 
										 std::vector<Particle >& particle_vec,
                                         const std::vector<Particle>& boundary_particle, 
                                         const float& time_step) 
{
	if(num_neigh == 0)
		return;
	glm::vec3 update, gradient_spiky;
	Particle tempPar;
	const int& vecSize = particle_vec.size();
	float unit;
	for( int i = 0; i < num_neigh; i++)
	{
		if (neighbor[i] < vecSize)
		{
			tempPar = particle_vec[neighbor[i]];
			if(tempPar.density>1.0e-3)
				unit = tempPar.mass/tempPar.density;
			else
				unit = tempPar.mass/rest_density;
			gradient_spiky = -Kernel::W_spiky_gradient( position - tempPar.position, radius);
			update+=(glm::length(tempPar.vorticity)-glm::length(vorticity))*unit*gradient_spiky;
		}
		else
		{
			tempPar = boundary_particle[neighbor[i]-vecSize];
			unit = tempPar.mass/rest_density;
			gradient_spiky = -Kernel::W_spiky_gradient( position - tempPar.position, radius);
			update+=(-glm::length(vorticity))*tempPar.mass/rest_density*gradient_spiky;
		}
	}
	if( glm::length( update ) != 0)
	{
		update /= glm::length(update);
		glm::vec3 result =  glm::cross(update, vorticity);
		result *= vorticity_epsilon/*time_step*/;
		if(density>1.0e-3)
			result /= density;
		else
			result /= rest_density;
		//velocity+=result;
		acceleration+=result;
	}
	return;
}


void Particle::collide_boundary_trivial(const Grid& grid)
{
	int tag[3];
	if (!stay_in_grid(position, grid, tag))
	{
		if (position.x <= grid.x_range.x+ ERROR_COLLISION)//penetrate from left x
		{
			position.x = grid.x_range.x + ERROR_COLLISION;
		}
		else if (position.x >= grid.x_range.y- ERROR_COLLISION)
		{
			position.x = grid.x_range.y - ERROR_COLLISION;//penetrate from right x
		}


		if (position.y <= grid.y_range.x+ ERROR_COLLISION)
		{
			position.y = grid.y_range.x + ERROR_COLLISION;//penetrate from left y

		}
		else if (position.y >= grid.y_range.y- ERROR_COLLISION)
		{
			position.y = grid.y_range.y - ERROR_COLLISION;//penetrate from right y
		}


		if (position.z <= grid.z_range.x+ ERROR_COLLISION) //penetrate from z bottom
		{
			position.z = grid.z_range.x + ERROR_COLLISION;
		}
		else if (position.z >= grid.z_range.y- ERROR_COLLISION)//penetrate from z top
		{
			position.z = grid.z_range.y - ERROR_COLLISION;
		}
	}
}

void Particle::boundaryClamp(const Grid& grid)
{
	const float& ERROR = 0.1*radius;
	if(position.x < grid.x_range.x+ERROR)
		position.x = grid.x_range.x+ERROR;
	else if(position.x > grid.x_range.y-ERROR)
		position.x = grid.x_range.y-ERROR;

	if(position.y < grid.y_range.x+ERROR)
		position.y = grid.y_range.x+ERROR;
	else if(position.y > grid.y_range.y-ERROR)
		position.y = grid.y_range.y-ERROR;

	if(position.z < grid.z_range.x+ERROR)
		position.z = grid.z_range.x+ERROR;
	else if(position.z > grid.z_range.y-ERROR)
		position.z = grid.z_range.y-ERROR;
}



const bool stay_in_grid(const glm::vec3& position, 
					    const Grid& grid, 
					    int tag[3]) // Judge whether the particle lies on the rigion part
{
	tag[0] = stay_in_interval(position.x, grid.x_range);
	tag[1] = stay_in_interval(position.y, grid.y_range);
	tag[2] = stay_in_interval(position.z, grid.z_range);
	if(tag[0]>0 && tag[1]>0 && tag[2]>0)
		return true;		
	else
		return false;
}



const int stay_in_interval(const float var, 
						   const glm::vec2& range)
{
	if( var > range.x && var < range.y )
	{
		if ((var - range.x) < (range.y - var))
			return 1; //closer to left limit
		else
			return 2; //closer to right limit
	}
	else if (var <= range.x)
	{
		return -1;
	}
	else if (var >= range.y)
	{
		return -2;
	}
}