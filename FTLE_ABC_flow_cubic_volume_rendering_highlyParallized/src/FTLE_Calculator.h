#pragma once

#include "Block.h"
#include "ABC_flow.h"
#include <vector>
#include <cassert>
#include <iostream>
#include <float.h>

#ifdef __APPLE__
	#include <Eigen/Eigen/Dense>
#elif __linux__
	#include <eigen3/Eigen/Dense>
#else
	error "Unknown compiler"
#endif

#include <sstream>
#include <climits>
#include <fstream>
using namespace Eigen;
using namespace std;

class FTLE_Calculator
{
public:

	// default constructor
	FTLE_Calculator(void);

	// destructor
	~FTLE_Calculator(void);

	// compute FTLE value based on given information
	void get_FTLE_value(float **data_information, const int& FRAME, const int& PARTICLE, const int& T,
									 const float& time_step, const int& choice, float readme[][2]);

private:

	// compute absolute distance between two vectors
	const float distance(float *a, float *b);

	// release neighbor vector for all particles
	void release_neighbor();

	// assign all particle positions into the decomposed domain for neighbor searching 
	void assigned_grid(float **data_information, const int& frame, const int& PARTICLE);

	// find neighbor index for a specific particle index of a specific frame number
	void find_neigbhor_index(float **data_information, const int& number, float& radius);

	// find neighbor and compute the searching radius based on grid size
	void find_neighbor(float **data_information, const int& PARTICLE);

	// compute FTLE based on neighbor information
	void compute_FTLE(float **data_information, const int& PARTICLE, const int& T, 
		              const float& time_step,const int& choice, const int& frame);

	// block grids for the domain decomposition
	std::vector< Hashing::Block> block_list;

	// neighbor information
	std::vector< std::vector<int > > neighbor;

	// generate vtk file for all the points
	void generate_full_vtk(float **data_information, const int& frame, const int& PARTICLE);

	// generate interior point vtk format
	void generate_interior_vtk(float **data_information, const int& frame, const int& PARTICLE);

	// parameter object to store grid size and grid
	Hashing::Parameter para;

	void getEachRealJacobian(float *position, const int& frame, const float& time_step, const int& choice);

	void getRealFTLE(float **data_information, const int& frame, const int& index, const int& T, 
		             const float& time_step, const int& choice);

	void computeEachLSF_ftle(float **data_information, const int& index, const int& T);

	void get_limit(float **data_information, const int& PARTICLE, float readme[][2]);

	void computeCentralDifference(float **data_information, const int& index, const int& T);

	void find_neigbhor_index(float **data_information, float *virtualPos, float& radius, std::vector<int>& virtualNeighbor);


};

// get max value among three given input
const float& get_max(const float& a, const float& b, const float& c);
const float getNorm(float *a);
const float getGaussian(float *virtualPos, float *virtualNeigh, const float &radius);