#include "Block.h"

// fix the grid to be 50* 50* 50
const int& SIZE = 60;

Hashing::Block::Block(void)
{
	inside = std::vector<int>();
}


Hashing::Block::~Block(void)
{
	inside.clear();
}


void Hashing::Parameter::compute_grid(float range[3][2])
{
	for (int i = 0; i < 3; i++)
	{
		RANGE[i][0] = range[i][0];
		RANGE[i][1] = range[i][1] + abs(range[i][1])*0.02; 
		//manually increase the right range by 0.02 so that the boundary value can be handled
		grid_size[i] = (RANGE[i][1] - RANGE[i][0])/SIZE;
		//grid[i] = ceil((range[i][1]-range[i][0])/grid_size[i]);
		grid[i] = SIZE;
	}
}

Hashing::Parameter::Parameter()
{

}

Hashing::Parameter::~Parameter()
{

}