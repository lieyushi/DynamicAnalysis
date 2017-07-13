#pragma once

#include <cmath>
#include <vector>
#include <cstddef>
#include <stdlib.h>

namespace Hashing
{
	class Block
	{
	public:
		Block(void);
		~Block(void);

		std::vector<int> inside;
	};


	class Parameter
	{
	public:
		float RANGE[3][2];
		float grid_size[3];		
		int grid[3];

		void compute_grid(float range[3][2]);
		Parameter();
		~Parameter();
	};
};

