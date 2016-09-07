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
		double RANGE[3][2];
		double grid_size[3];		
		int grid[3];

		void compute_grid(double range[3][2]);
		Parameter();
		~Parameter();
	};
};


