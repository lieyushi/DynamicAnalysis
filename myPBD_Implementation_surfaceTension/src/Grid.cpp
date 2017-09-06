#include "Grid.h"


Grid::Grid()
{
	x_range.x = (float)0.0;
	x_range.x = (float)0.0;
	y_range.x = (float)0.0;
	y_range.y = (float)0.0;
	z_range.x = (float)0.0;
	z_range.y = (float)0.0;
}

Grid::Grid(const float x[2], 
		   const float y[2], 
		   const float z[2])
{
	x_range.x = x[0];
	x_range.y = x[1];
	y_range.x = y[0];
	y_range.y = y[1];
	z_range.x = z[0];
	z_range.y = z[1];
}

Grid::~Grid()
{
	
}