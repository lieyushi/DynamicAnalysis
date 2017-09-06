#ifndef _SURFACETENSION_H_
#define _SURFACETENSION_H_

#include "Particle.h"
#include "Kernel.h"

#define surfaceConst 0.05

class SurfaceTension
{
public:

	static void addWCSPH_surfaceTension(const std::vector<Particle>& boundaryParticle, 
										std::vector<Particle>& fluidParticle,
										const float& time_step);

	static void addVersatle_surfaceTension(const std::vector<Particle>& boundaryParticle, 
										   std::vector<Particle>& fluidParticles,
										   const float& time_step);

	static void addHe_surfaceTension(const std::vector<Particle>& boundaryParticle, 
									 std::vector<Particle>& fluidParticle,
									 const float& time_step);

	static void noTension(const std::vector<Particle>& boundaryParticle, 
						  std::vector<Particle>& fluidParticle,
						  const float& time_step);

private:

	static void computeNormals(const std::vector<Particle>& boundaryParticle, 
							   std::vector<Particle>& fluidParticle);

	static void computeColor(const std::vector<Particle>& boundaryParticle, 
							 std::vector<Particle>& fluidParticle,
							 std::vector<float>& color);

	static void computeColorGradient(const std::vector<Particle>& boundaryParticle, 
									 std::vector<Particle>& fluidParticle,
							 		 const std::vector<float>& color, 
							 		 std::vector<float>& colorGrad);

	static void clampMaxSpeed(glm::vec3& velocity, const float& maxSpeed);
};

#endif