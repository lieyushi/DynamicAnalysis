#include "Particle.h"

const float ERROR_VELO = 1E-6;
const float ERROR_COLLISION = 0.02;
const float MIN = 400;
Particle::Particle(const float posi[3], const float& radius)
{
	mass = 0;
	position.x = posi[0]; position.y = posi[1]; position.z = posi[2]; 
	this->radius = radius;
	velocity.x = velocity.y = velocity.z = 0;
	vorticity.x = vorticity.y = vorticity.z = 0.0;
}


Particle::~Particle(void)
{

}

Particle::Particle()
{

}

Particle::Particle(const float& mass, const float posi[3], const float velo[3], const float& radius, const float force[3], const int& index)
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


const bool Particle::assigned_to_block( std::vector<Block>& block_list, 
	int *block_unit, const float& blockSize, const Grid& boundary)
{
	
	posi_index[0] = (int)floor((position.x - boundary.x_range.x)/blockSize);
	posi_index[1] = (int)floor((position.y - boundary.y_range.x)/blockSize);
	posi_index[2] = (int)floor((position.z - boundary.z_range.x)/blockSize); // calculating the index array of point based on grid

	/*assert(posi_index[0] >= 0 && posi_index[0] < block_unit[0]);
	assert(posi_index[1] >= 0 && posi_index[1] < block_unit[1]);
	assert(posi_index[2] >= 0 && posi_index[2] < block_unit[2]);*/

	if(!((posi_index[0] >= 0 && posi_index[0] < block_unit[0]) && (posi_index[1] >= 0 && posi_index[1] < block_unit[1])
	   && (posi_index[2] >= 0 && posi_index[2] < block_unit[2])))
	{
		return false;
	}

	const int& block_index = posi_index[0] + block_unit[0] * posi_index[1] + block_unit[0] * block_unit[1] * posi_index[2];	
	if(!(block_index >= 0 && block_index < block_list.size()))
		return false;
	block_list[block_index].add(index);	
	return true;	
}



void Particle::get_density( std::vector<Particle >& particle_vec, const std::vector<Particle>& boundary_vec ) // compute density from the neighborhood particles
{
	density = (float)0.0;
	/*if(num_neigh == 0)
	{
		std::cout << "Particle " << index << " has 0 density!" << std::endl;
	}*/

	for( int i = 0; i < num_neigh; i++)
	{
		const int& i_ = neighbor[i];
		if ( i_ < particle_vec.size())
		{
			const float& poly6 = Kernel::W_poly6(position-particle_vec[i_].position, radius);
			density += particle_vec[neighbor[i]].mass * poly6;
		}
		else
		{
			const float& poly6 = Kernel::W_poly6(position - boundary_vec
			[i_-particle_vec.size()].position, radius);
			density += boundary_vec[neighbor[i]-particle_vec.size()].mass * poly6;
		}
	}
}

void Particle::compute_factor_by_constraint(const float& constraint,
	const float& factor_epsilon, std::vector<Particle >& particle_vec, 
	const std::vector<Particle>& boundary_particle) 
{
	factor = 0.0;
	if(constraint == 0 || num_neigh == 0)
		return;
	
	float sum_constraint = 0.0;

	////for k = i case
	glm::vec3 temp;
	for(int i = 0; i < num_neigh; i++)
	{
		glm::vec3 spiky_gradient;
		if (neighbor[i] < particle_vec.size())
			spiky_gradient = -particle_vec[neighbor[i]].mass/rest_density*Kernel::W_spiky_gradient(position - particle_vec[neighbor[i]].position, radius);
		else
			spiky_gradient = -boundary_particle[neighbor[i] - particle_vec.size()].mass/rest_density*
			Kernel::W_spiky_gradient(position - boundary_particle[neighbor[i] - particle_vec.size()].position, radius);
		temp += spiky_gradient; 
		sum_constraint += pow(glm::length(spiky_gradient),2);  
	}
	sum_constraint += pow(glm::length(temp), 2);
	factor = - constraint/(sum_constraint + factor_epsilon);
}


void Particle::compute_factor(const float& factor_epsilon, std::vector<Particle >& particle_vec, const std::vector<Particle>& boundary_particle)// compute the scaling factor (also called artificial pressure term) with consideration of collision constraint
{
	const float& c_i = std::max(float(density/rest_density - 1), float(0.0));
	//const float& c_i =  density/rest_density - 1.0;
	compute_factor_by_constraint(c_i, factor_epsilon, particle_vec, boundary_particle);
}


void Particle::compute_position_update_by_factor( 
	std::vector<Particle >& particle_vec, 
	const std::vector<Particle>& boundary_particle )//get position update value from artificial pressure term directly
{
	delta_position = glm::vec3(0);
	if(num_neigh == 0)
		return;
	const float delta_q = (float)DELTA * radius;
	for(int i = 0; i < num_neigh; i++)
	{
		float upper, artificial;
		const float& lower = Kernel::W_poly6_scalar(delta_q, radius);		
		glm::vec3 grad;
		float multi;
		if (neighbor[i] < particle_vec.size())
		{
			upper = Kernel::W_poly6(position - particle_vec[neighbor[i]].position, radius);
			grad = particle_vec[neighbor[i]].mass/rest_density*Kernel::W_spiky_gradient(position - particle_vec[neighbor[i]].position, radius); //The derivative should be under i
			artificial = - (float)ART_COEFFICIENT * pow(upper / lower, ART_EXPONENTIAL);
			//if(density >= MIN)
				multi = factor + particle_vec[neighbor[i]].factor /*+ artificial*/;	
			//else
				//multi = factor + particle_vec[neighbor[i]].factor + artificial;			
		}
		else
		{
			upper = Kernel::W_poly6(position - boundary_particle[neighbor[i] - particle_vec.size()].position, radius);
			artificial = - (float)ART_COEFFICIENT * pow(upper / lower, ART_EXPONENTIAL);
			grad = boundary_particle[neighbor[i]-particle_vec.size()].mass/rest_density*
			Kernel::W_spiky_gradient(position - boundary_particle[neighbor[i] - particle_vec.size()].position, radius); //The derivative should be under i
			//if(density >= MIN)
				multi = factor /*+ artificial*/;
			//else
				//multi = factor + artificial;
			
		}

		delta_position.x += grad.x * multi;
		delta_position.y += grad.y * multi;
		delta_position.z += grad.z * multi;		
	}
}

void Particle::compute_vorticity( std::vector<Particle >& particle_vec, 
								const std::vector<Particle>& boundary_particle)     //compute vorticity term which forms a force 
{
	vorticity = glm::vec3(0.0);
	if(num_neigh == 0)
		return;
	for(int i = 0; i < num_neigh; i++)
	{
		if (neighbor[i] < particle_vec.size())
		{
			//if(particle_vec[neighbor[i]].density < 1.0)
			//{
			//	const glm::vec3& spiky_gradient = - /*particle_vec[neighbor[i]].mass/particle_vec[neighbor[i]].density*/
			//	Kernel::W_spiky_gradient(position - particle_vec[neighbor[i]].position, radius);
			//	vorticity += glm::cross(  particle_vec[neighbor[i]].velocity - velocity, spiky_gradient);
			//}
			//else
			//{
				const glm::vec3& spiky_gradient = - particle_vec[neighbor[i]].mass/particle_vec[neighbor[i]].density*
				Kernel::W_spiky_gradient(position - particle_vec[neighbor[i]].position, radius);
				vorticity += glm::cross(  particle_vec[neighbor[i]].velocity - velocity, spiky_gradient);
			//}
		
		}		
		/*else
		{
			const glm::vec3& spiky_gradient = - boundary_particle[neighbor[i]-particle_vec.size()].mass/rest_density*
			Kernel::W_spiky_gradient(position - boundary_particle[neighbor[i]-particle_vec.size()].position, radius);
			vorticity += glm::cross(  boundary_particle[neighbor[i]-particle_vec.size()].velocity - velocity, spiky_gradient);
		}*/
	}
	/*if(abs(vorticity.x) < ERROR_VELO)
		vorticity.x = 0;
	if(abs(vorticity.y) < ERROR_VELO)
		vorticity.y = 0;
	if(abs(vorticity.z) < ERROR_VELO)
		vorticity.z = 0;*/
	assert(!(isnan(vorticity.x)) && !(isnan(vorticity.y)) &&!(isnan(vorticity.z)));
}


void Particle::get_velocity_by_vorticity(const float& vorticity_epsilon,
	std::vector<Particle >& particle_vec, const std::vector<Particle>& boundary_particle, const float& time_step)   // produce velocity variation towards the particle velocity
{
	if(num_neigh == 0)
		return;
	glm::vec3 update;
	for( int i = 0; i < num_neigh; i++)
	{
		if (neighbor[i] < particle_vec.size())
		{
			const Particle& particle = particle_vec[neighbor[i]];
			//if(particle.density < 1.0)
			//{
			//	const glm::vec3& gradient_spiky = Kernel::W_spiky_gradient( position - particle.position, radius);
			//	update += (glm::length(particle.vorticity) - glm::length(vorticity)) /* particle_vec[neighbor[i]].mass/particle_vec[neighbor[i]].density */* gradient_spiky;
			//}
			//else
			//{
				const glm::vec3& gradient_spiky = Kernel::W_spiky_gradient( position - particle_vec[neighbor[i]].position, radius);
				update += (glm::length(particle.vorticity) - glm::length(vorticity)) * particle_vec[neighbor[i]].mass/particle_vec[neighbor[i]].density * gradient_spiky;
			//}
		}
		/*else{
			const glm::vec3& gradient_spiky = Kernel::W_spiky_gradient( position - boundary_particle[neighbor[i]-particle_vec.size()].position, radius);
			update.x += (glm::length(boundary_particle[neighbor[i]-particle_vec.size()].vorticity) - glm::length(vorticity)) * boundary_particle[neighbor[i]-particle_vec.size()].mass/rest_density * gradient_spiky.x;
			update.y += (glm::length(boundary_particle[neighbor[i]-particle_vec.size()].vorticity) - glm::length(vorticity)) * boundary_particle[neighbor[i]-particle_vec.size()].mass/rest_density * gradient_spiky.y;
			update.z += (glm::length(boundary_particle[neighbor[i]-particle_vec.size()].vorticity) - glm::length(vorticity)) * boundary_particle[neighbor[i]-particle_vec.size()].mass/rest_density * gradient_spiky.z;
		}*/
		
	}
	if( glm::length( update ) > 1.0e-6)
	{
		update /= glm::length(update);
		glm::vec3 result =  glm::cross(update, vorticity);
		result *= vorticity_epsilon*time_step;
		velocity += result;
	}
	return;
	/*------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		The second method is to calculate the gradient of norm of vorticity by least square of neighbor particle vorticites values------------------------------------------------------------------*/
	//glm::vec3 x_summation;
	//float x_norm = 0;
	//for( int i = 0; i < num_neigh; i++)
	//{
	//	x_summation += glm::length(particle_vec[neighbor[i]].vorticity) * ( particle_vec[neighbor[i]].position - position );
	//	x_norm += pow(glm::length(particle_vec[neighbor[i]].position - position) , 2);
	//}
	//glm::vec3 gradient = x_summation / x_norm;
	//if( glm::length(gradient) != 0)
	//	gradient /= glm::length(gradient);
	//else
	//	return glm::vec3(0.0);
	//glm::vec3& result =  glm::cross(gradient, vorticity);
	//result *= vorticity_epsilon;
	//result /= mass;
	//return result;


/*------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    The third method is to calculate the gradient of norm of vorticity from the definition formula of voriticy, we can take gradient kernel as function of position -----------------------------*/
	//glm::highp_dmat3x3 gradient; 
	//for( int i = 0; i < num_neigh; i++)//This calculates the respective gradient of spiky gradient of neighboring particles for a particle
	//{ 
	//	glm::vec3 x_gradient, y_gradient, z_gradient;//particle is a neighboring particle
	//	float total_norm = 0; 
	//	for( int j = 0; j < particle_vec[ neighbor[i] ].num_neigh; ++ j)
	//	{
	//		glm::vec3 spiky_graident = - Kernel::W_spiky_gradient( particle_vec[neighbor[i]].position - particle_vec[particle_vec[neighbor[i]].neighbor[j]].position, radius );
	//		total_norm += pow(glm::length(particle_vec[particle_vec[neighbor[i]].neighbor[j]].position - particle_vec[neighbor[i]].position), 2);
	//		x_gradient += spiky_graident.x * ( particle_vec[particle_vec[neighbor[i]].neighbor[j]].position - particle_vec[neighbor[i]].position );
	//		y_gradient += spiky_graident.y * ( particle_vec[particle_vec[neighbor[i]].neighbor[j]].position - particle_vec[neighbor[i]].position );
	//		z_gradient += spiky_graident.z * ( particle_vec[particle_vec[neighbor[i]].neighbor[j]].position - particle_vec[neighbor[i]].position );
	//	}
	//	glm::vec3 x_least_square = - glm::cross( particle_vec[neighbor[i]].velocity - velocity, x_gradient/total_norm );
	//	glm::vec3 y_least_square = - glm::cross( particle_vec[neighbor[i]].velocity - velocity, y_gradient/total_norm );
	//	glm::vec3 z_least_square = - glm::cross( particle_vec[neighbor[i]].velocity - velocity, z_gradient/total_norm );
	//	gradient += glm::transpose( glm::highp_dmat3x3( x_least_square, y_least_square, z_least_square ));
	//}
	//glm::vec3 gradient_of_norm = gradient * vorticity / glm::length( vorticity );
	//if( glm::length( gradient_of_norm ) != 0)
	//	gradient_of_norm /= glm::length( gradient_of_norm );
	//glm::vec3 result = glm::cross( gradient_of_norm, vorticity ) * vorticity_epsilon/mass;
	//return result;
}


void Particle::compute_velocity_by_viscosity( std::vector<Particle >& particle_vec, const std::vector<Particle>& boundary_particle )     
{
	if(num_neigh == 0)
		return;
	glm::vec3 change_XSPH; 
	for(int i = 0; i < num_neigh; i++)
	{
		const Particle& particle = particle_vec[neighbor[i]];
		if (neighbor[i] < particle_vec.size())
		{
			//if (particle.density < 1.0)
			//{
			//	const glm::vec3& velocity_dist = /*particle_vec[neighbor[i]].mass/particle_vec[neighbor[i]].density*/
			//										(particle_vec[neighbor[i]].velocity - velocity);
			//	change_XSPH +=  Kernel::W_poly6(position - particle_vec[neighbor[i]].position, radius) * velocity_dist;
			//}
			//else
			//{
				const glm::vec3& velocity_dist = particle_vec[neighbor[i]].mass/particle_vec[neighbor[i]].density*
													(particle_vec[neighbor[i]].velocity - velocity);
				change_XSPH +=  Kernel::W_poly6(position - particle_vec[neighbor[i]].position, radius) * velocity_dist;
			//}
		}
		/*else
		{
			const glm::vec3& velocity_dist = boundary_particle[neighbor[i]-particle_vec.size()].mass/rest_density*
													(boundary_particle[neighbor[i]-particle_vec.size()].velocity - velocity);
			change_XSPH +=  Kernel::W_poly6(position - boundary_particle[neighbor[i]-particle_vec.size()].position, radius) * velocity_dist;
		}*/
		
	}
	change_XSPH *= (float)COHENRENT;
	velocity += change_XSPH;

	/*if(abs(velocity.x) < ERROR_VELO)
		velocity.x = 0;
	if(abs(velocity.y) < ERROR_VELO)
		velocity.y = 0;
	if(abs(velocity.z) < ERROR_VELO)
		velocity.z = 0;*/
	if(isnan(velocity.x) || isnan(velocity.y) || isnan(velocity.z))
	{
		std::cout << "Particle " << index << " velocity has inf value!" << std::endl;
		exit(-1);
	}
}


void Particle::boundaryClamp(const Grid& grid)
{
	if(position.x < grid.x_range.x)
		position.x = grid.x_range.x;
	else if(position.x > grid.x_range.y)
		position.x = grid.x_range.y;

	if(position.y < grid.y_range.x)
		position.y = grid.y_range.x;
	else if(position.y > grid.y_range.y)
		position.y = grid.y_range.y;

	if(position.z < grid.z_range.x)
		position.z = grid.z_range.x;
	else if(position.z > grid.z_range.y)
		position.z = grid.z_range.y;
}


void Particle::collide_boundary_trivial(const Grid& grid)
{
	if (!stay_in_grid(position, grid))
	{
		if (position.x <= grid.x_range.x)//penetrate from left x
		{
			position.x = grid.x_range.x + ERROR_COLLISION;
			velocity.x = 0.0;
		}
		else if (position.x >= grid.x_range.y)
		{
			position.x = grid.x_range.y - ERROR_COLLISION;//penetrate from right x
			velocity.x = 0.0;
		}


		if (position.y <= grid.y_range.x)
		{
			position.y = grid.y_range.x + ERROR_COLLISION;//penetrate from left y
			velocity.y = 0.0;

		}
		else if (position.y >= grid.y_range.y)
		{
			position.y = grid.y_range.y - ERROR_COLLISION;//penetrate from right y
			velocity.y = 0.0;
		}


		if (position.z <= grid.z_range.x) //penetrate from z bottom
		{
			position.z = grid.z_range.x + ERROR_COLLISION;
			velocity.z = 0.0;
		}
		else if (position.z >= grid.z_range.y)//penetrate from z top
		{
			position.z = grid.z_range.y - ERROR_COLLISION;
			velocity.z = 0.0;
		}

		//velocity.x = velocity.y = velocity.z = 0.0;
		std::cout << "Particle " << index << " launches trivial technique!" << std::endl;
	}
}

void Particle::collide_boundary_bouncing(const Grid& grid)
{
	if (!stay_in_grid(position, grid))
	{
		if (position.x <= grid.x_range.x)//penetrate from left x
		{
			position.x = 2.0*grid.x_range.x - position.x;
			velocity.x = 0.0;
		}
		else if (position.x >= grid.x_range.y)
		{
			position.x = 2.0*grid.x_range.y - position.x;//penetrate from right x
			velocity.x = 0.0;
		}


		if (position.y <= grid.y_range.x)
		{
			position.y = 2.0*grid.y_range.x - position.y;//penetrate from left y
			velocity.y = 0.0;
		}
		else if (position.y >= grid.y_range.y)
		{
			position.y = 2.0*grid.y_range.y - position.y;//penetrate from right y
			velocity.y = 0.0;
		}


		if (position.z <= grid.z_range.x) //penetrate from z bottom
		{
			position.z = 2.0*grid.z_range.x - position.z;
			velocity.z = 0.0;
		}
		else if (position.z >= grid.z_range.y)//penetrate from z top
		{
			position.z = 2.0*grid.z_range.y - position.z;
			velocity.z = 0.0;
		}

		//velocity.x = velocity.y = velocity.z = 0.0;

		std::cout << "Particle " << index << " launches bouncing technique!" << std::endl;
	}
}



const bool stay_in_grid(const glm::vec3& position, const Grid& grid) // Judge whether the particle lies on the rigion part
{
	if(stay_in_interval(position.x, grid.x_range) >0 && stay_in_interval(position.y, grid.y_range) >0 && stay_in_interval(position.z, grid.z_range) >0)
		return true;		
	else
		return false;
}


const int stay_in_interval(const float var, const glm::vec2& range)
{
	if( var > range.x && var < range.y )
	{
		if ((var - range.x) < (range.y - var))
			return 1; //closer to left limit
		else
			return 2; //closer to right limit
	}
	else if (var < range.x)
	{
		return -1;
	}
	else if (var > range.y)
	{
		return -2;
	}
}