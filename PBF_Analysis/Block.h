#pragma once

#include <cmath>
#include <vector>
#include <cstddef>
#include <stdlib.h>
#include <cassert>

namespace Hashing
{
	class Block
	{
	public:
		Block(void);
		~Block(void);
		Block(const Block&);
		void add(const int& index);

		int* inside;
		static int max_size;
		int size;
	};


	class Parameter
	{
	public:
		float RANGE[3][2];
		float grid_size[3]; //grid length, width, and height		
		int grid[3]; //grid number

		void compute_grid(const float range[3][2], const float& radius);
		Parameter();
		~Parameter();
	};
};


