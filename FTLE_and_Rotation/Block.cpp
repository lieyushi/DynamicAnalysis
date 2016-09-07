#include "Block.h"

// fix the grid to be 50* 50* 50
int Hashing::Block::max_size = 50;

Hashing::Block::Block(void)
{
	inside = new int[max_size];
	size = 0;
}


Hashing::Block::~Block(void)
{
	delete[] inside;
	size = 0;
}


void Hashing::Block::add(const int& index)
{
	assert(size < max_size);
	inside[size++] = index;
}

/* This is to define a copy constructor for Block to create pointer */
Hashing::Block::Block(const Block&)  
{
	inside = new int[max_size];
	size = 0;
}

void Hashing::Parameter::compute_grid(const double range[3][2], const double& radius)
{
	for (int i = 0; i < 3; i++)
	{
		RANGE[i][0] = range[i][0];
		RANGE[i][1] = range[i][1]; 
		grid_size[i] = radius;
		grid[i] = ceil((RANGE[i][1] - RANGE[i][0])/grid_size[i]) + 1;
	}
}

Hashing::Parameter::Parameter()
{

}

Hashing::Parameter::~Parameter()
{

}