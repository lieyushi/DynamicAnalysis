#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <cmath>

const float pi = 3.1415926;

class Kernel
{
public:
	static const float W_poly6(const glm::vec3& r, const float& radius);
	static const float W_poly6(const float& r, const float& radius);
    static const float W_spiky(const glm::vec3& r, const float& radius);
	static const float W_spiky(const float& dist, const float& radius);
	static const glm::vec3 W_poly6_gradient(const glm::vec3& r, const float& radius);
	static const glm::vec3 W_spiky_gradient(const glm::vec3& r, const float& radius);
	static const float W_cubic(const glm::vec3& r, const float& radius);
	static const float W_cubic(const float& r, const float& radius);
	static const glm::vec3 W_cubic_gradient(const glm::vec3& r, const float& radius);
	static const float W_cubic_laplacian(const glm::vec3& r, const float& radius);
	static const float W_cohesion(const glm::vec3& r, const float& radius);
};
