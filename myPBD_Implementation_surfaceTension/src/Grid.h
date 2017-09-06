#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
class Grid
{
public:
	Grid(const float x[2], 
		 const float y[2], 
		 const float z[2]);
	Grid();
	~Grid();

	glm::vec2 x_range;
	glm::vec2 y_range;
	glm::vec2 z_range;
};

