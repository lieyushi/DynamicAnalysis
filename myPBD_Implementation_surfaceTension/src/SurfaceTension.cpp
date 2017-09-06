#include "SurfaceTension.h"

void SurfaceTension::addWCSPH_surfaceTension(const std::vector<Particle>& boundaryParticle, std::vector<Particle>& fluidParticle,
											 const float& time_step)
{
	const int& fluidSize = fluidParticle.size();
	const float& diameter = fluidParticle[0].radius/2.0;
	const float& maxSpeed = 0.5*fluidParticle[0].radius/time_step;

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < fluidSize; ++i)
	{
		Particle& target = fluidParticle[i];
		const glm::vec3& xi = target.position;
		glm::vec3 &accel = target.acceleration;

		const std::vector<int>& targetNeigh = target.neighbor;
		for (int j = 0; j < targetNeigh.size(); ++j)
		{
			const int& neighborIndex = targetNeigh[j];
			Particle neighborParticle;
			if(neighborIndex < fluidSize)
				neighborParticle = fluidParticle[neighborIndex];
			else
				neighborParticle = boundaryParticle[neighborIndex-fluidSize];

			const glm::vec3& xj = neighborParticle.position;
			const glm::vec3& xij = xi-xj;
			const float& dist = glm::length(xij);
			if(dist > diameter)
				accel -= (float)surfaceConst/target.mass*neighborParticle.mass*xij*Kernel::W_poly6(xij, target.radius);
			else
				accel -= (float)surfaceConst/target.mass*neighborParticle.mass*xij*Kernel::W_poly6(glm::vec3(diameter,0,0), target.radius);
		}
		target.original_position = target.position;
		target.velocity += target.acceleration*time_step;
		clampMaxSpeed(target.velocity, maxSpeed);
		target.acceleration = glm::vec3(0);
	}
}


void SurfaceTension::addVersatle_surfaceTension(const std::vector<Particle>& boundaryParticle, std::vector<Particle>& fluidParticle,
												const float& time_step)
{
	computeNormals(boundaryParticle, fluidParticle);
	const float& maxSpeed = 0.5*fluidParticle[0].radius/time_step;

	const int& fluidSize = fluidParticle.size();
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < fluidSize; ++i)
	{
		Particle& target = fluidParticle[i];
		glm::vec3& accel = target.acceleration;
		const glm::vec3& x_i = target.position;
		const float& targetDens = target.density;
		const glm::vec3& n_i = target.normal;
		int tempIndex;
		for (int j = 0; j < target.num_neigh; ++j)
		{
			tempIndex = target.neighbor[j];
			if(tempIndex<fluidSize)
			{
				const Particle& neighborParti = fluidParticle[tempIndex];
				const float& K_ij = 2.0*rest_density/(targetDens+neighborParti.density);

				glm::vec3 tempAcc(0);
				const glm::vec3& x_j = neighborParti.position;
				glm::vec3 xixj = x_i-x_j;
				const float& xixj_length = glm::length(xixj);
				if(xixj_length>1.0e-5)
				{
					xixj/=xixj_length;
					tempAcc -= (float)surfaceConst*neighborParti.mass*xixj*Kernel::W_cohesion(x_i-x_j, target.radius);
				}

				const glm::vec3& n_j = neighborParti.normal;
				tempAcc -= (float)surfaceConst*target.radius*(n_i-n_j);

				accel += K_ij*tempAcc;
			}
			else
			{
				const Particle& neighborParti = boundaryParticle[tempIndex-fluidSize];
				const glm::vec3& x_j = neighborParti.position;
				glm::vec3 xixj = x_i-x_j;
				const float& xixj_length = glm::length(xixj);
				if(xixj_length>1.0e-5)
				{
					xixj/=xixj_length;

					accel -= (float)surfaceConst*neighborParti.mass*xixj*Kernel::W_cohesion(x_i-x_j, target.radius);
				}
			}
		}

		target.original_position = target.position;
		target.velocity += target.acceleration*time_step;
		clampMaxSpeed(target.velocity, maxSpeed);
		target.acceleration = glm::vec3(0);
	}
}


void SurfaceTension::addHe_surfaceTension(const std::vector<Particle>& boundaryParticle, std::vector<Particle>& fluidParticle, 
										  const float& time_step)
{
	std::vector<float> color(fluidParticle.size(),0);
	std::vector<float> color_grad(fluidParticle.size(),0);
	const float& maxSpeed = 0.5*fluidParticle[0].radius/time_step;

	computeColor(boundaryParticle, fluidParticle, color);
	computeColorGradient(boundaryParticle, fluidParticle, color, color_grad);

	const int& fluidSize = fluidParticle.size();
#pragma omp parallel for schedule(dynamic) num_threads(8)	
	for (int i = 0; i < fluidSize; ++i)
	{
		Particle& target = fluidParticle[i];
		float factor = (float)(0.25*surfaceConst);
		if(target.density>1.0e-3)
			factor/=target.density;
		else
			factor/=rest_density;
		const float& gradC_i = color_grad[i];
		const glm::vec3& x_i = target.position;
		glm::vec3& accel = target.acceleration;
		int neighborIndex;
		float unit;
		for (int j = 0; j < target.num_neigh; ++j)
		{
			neighborIndex = target.neighbor[j];
			if(neighborIndex<fluidSize)
			{
				const Particle& neighborParticle = fluidParticle[neighborIndex];
				if(neighborParticle.density>=1.0e-3)
					unit = neighborParticle.mass/neighborParticle.density;
				else
					unit = neighborParticle.mass/rest_density;
				accel += factor*unit*(gradC_i+color_grad[neighborIndex])
						 *Kernel::W_spiky_gradient(x_i-neighborParticle.position, target.radius);
			}
			else
			{
				const Particle& neighborParticle = boundaryParticle[neighborIndex-fluidSize];
				unit = neighborParticle.mass/rest_density;
				accel += factor*unit*gradC_i*
						 Kernel::W_spiky_gradient(x_i-neighborParticle.position,target.radius);
			}
		}

		target.original_position = target.position;
		target.velocity += target.acceleration*time_step;
		clampMaxSpeed(target.velocity, maxSpeed);
		target.acceleration = glm::vec3(0);
	}
}


void SurfaceTension::computeNormals(const std::vector<Particle>& boundaryParticle, std::vector<Particle>& fluidParticle)
{
	const int& fluidSize = fluidParticle.size();
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < fluidSize; ++i)
	{
		Particle& target = fluidParticle[i];
		glm::vec3& normal = target.normal;
		normal = glm::vec3(0);

		const glm::vec3& targetPos = target.position;
		int tempSize;
		Particle tempPart;
		float unit;
		for (int j = 0; j < target.num_neigh; ++j)
		{
			tempSize = target.neighbor[j];
			if(tempSize<fluidSize)
			{
				tempPart = fluidParticle[tempSize];
				if(tempPart.density>1.0e-3)
					unit = tempPart.mass/tempPart.density;
				else
					unit = tempPart.mass/rest_density;
				normal+=unit*Kernel::W_spiky_gradient(targetPos-tempPart.position, target.radius);
			}
		}
		normal *= target.radius;
	}
}


void SurfaceTension::computeColor(const std::vector<Particle>& boundaryParticle, std::vector<Particle>& fluidParticle,
							 std::vector<float>& color)
{
	const int& fluidSize = fluidParticle.size();
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < fluidSize; i++)
	{
		Particle& target = fluidParticle[i];
		const glm::vec3& x_i = target.position;
		float& c_i = color[i];
		int neighborIndex;
		float unit;
		for (int j = 0; j < target.num_neigh; ++j)
		{
			neighborIndex = target.neighbor[j];
			if(neighborIndex<fluidSize)
			{
				const Particle& neighborParticle = fluidParticle[neighborIndex];
				if(neighborParticle.density>1.0e-3)
					unit = neighborParticle.mass/neighborParticle.density;
				else
					unit = neighborParticle.mass/rest_density;
				c_i += unit*Kernel::W_poly6(x_i-neighborParticle.position, target.radius);
			}
			else
			{
				const Particle& neighborParticle = boundaryParticle[neighborIndex-fluidSize];
				c_i += neighborParticle.mass/rest_density*Kernel::W_poly6(x_i-neighborParticle.position, target.radius);
			}
		}
	}
}
	
void SurfaceTension::computeColorGradient(const std::vector<Particle>& boundaryParticle, std::vector<Particle>& fluidParticle,
							 const std::vector<float>& color, std::vector<float>& colorGrad)
{
	const int& fluidSize = fluidParticle.size();
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < fluidSize; i++)
	{
		Particle& target = fluidParticle[i];
		const glm::vec3& x_i = target.position;
		glm::vec3 gradC(0.0);
		float unit;
		//const float& targetDens = target.density;
		int neighborIndex;
		for (int j = 0; j < target.num_neigh; ++j)
		{
			neighborIndex = target.neighbor[j];
			if(neighborIndex<fluidSize)
			{
				const Particle& neighborParticle = fluidParticle[neighborIndex];
				if(neighborParticle.density>1.0e-3)
					unit = neighborParticle.mass/neighborParticle.density;
				else
					unit = neighborParticle.mass/rest_density;
				gradC+=unit*color[neighborIndex]*Kernel::W_spiky_gradient
					   (x_i-neighborParticle.position, target.radius);
			}
		}
		if(color[i]>1.0e-6)
			gradC *=(1.0/color[i]);
		else
			gradC *=1.0e3;
		colorGrad[i] = glm::dot(gradC,gradC);
	}	
}


void SurfaceTension::noTension(const std::vector<Particle>& boundaryParticle, std::vector<Particle>& fluidParticle,
							   const float& time_step)
{
	const int& fluidSize = fluidParticle.size();
	const float& maxSpeed = 0.5*fluidParticle[0].radius/time_step;
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < fluidSize; ++i)
	{
		Particle& target = fluidParticle[i];
		target.original_position = target.position;
		target.velocity += target.acceleration*time_step;
		clampMaxSpeed(target.velocity, maxSpeed);
		target.acceleration = glm::vec3(0);
	}
}


void SurfaceTension::clampMaxSpeed(glm::vec3& velocity, const float& maxSpeed)
{
	const float& speed = glm::length(velocity);
	if(speed>maxSpeed)
	{
		velocity*=maxSpeed/speed;
	}
}