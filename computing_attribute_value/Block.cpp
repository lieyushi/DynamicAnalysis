#include "Block.h"

// fix the grid to be 50* 50* 50
const int& SIZE = 50;
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

void Hashing::Parameter::compute_grid(double range[3][2])
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