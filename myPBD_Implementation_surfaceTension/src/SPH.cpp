#include "SPH.h"
#include <cmath>
#include<fstream>
#include<iostream>
#include<sstream>
#include<string>
using namespace std;

const float LEAST_VALUE = 0.0000000001;

SPH::SPH(void)
{

}

SPH::~SPH(void)
{
	cout << "SPH destructor invoked!" << endl;
}

SPH::SPH(const string& file_)
{
	const int even[3] = {60, 50, 65};
	fluid_number = even[0]*even[1]*even[2];
	iteration = 4;
	time_step = 0.016;
	const float& interval = 1.0;
	const float& supportRadius = 2.0*interval;
	blockSize = interval;
	const float& mass = 0.8*rest_density*pow(interval,3.0);
	const float external_foce[] = {0.0, 0.0, -gravity*mass};

	std::cout << "Distance among particles are " << interval << " " << interval << " " << interval << std::endl;
	
	const float x[2] = {-supportRadius, interval*even[0]*(float)2.2};
	const float y[2] = {-supportRadius, interval*even[1]*(float)1.2};
	const float z[2] = {-supportRadius, interval*even[2]*(float)2.2};

	boundary = Grid(x, y, z);

	std::cout << "Boudnary domain is: [" << boundary.x_range.x << ", " << boundary.x_range.y << "] X [" 
										 << boundary.y_range.x << ", " << boundary.y_range.y << "] X ["
										 << boundary.z_range.x << ", " << boundary.z_range.y << "]!" << std::endl;

	block_unit[0] = ceil((boundary.x_range.y - boundary.x_range.x)/blockSize) + 1;
	block_unit[1] = ceil((boundary.y_range.y - boundary.y_range.x)/blockSize) + 1;
	block_unit[2] = ceil((boundary.z_range.y - boundary.z_range.x)/blockSize) + 1; // compute the block unit
	block_number = block_unit[0]*block_unit[1]*block_unit[2];

	particle_list = std::vector<Particle>(fluid_number);
	ifstream fin(file_.c_str(), ios::in);
	if (fin.fail())
	{
		cout << "Error creating the files!" << endl;
		exit(-1);
	}

	float velo[3];
	float a;
	for(int i = 0; i < fluid_number; i++)
	{
		float posi[3];
		float velo[3];
		fin >> i >> posi[0] >> posi[1] >> posi[2] >> velo[0] >> velo[1] >> velo[2] >> a;
		particle_list[i] = Particle(mass, posi, velo, supportRadius, external_foce, i);
	}
	std::cout << "Fluid particles distribution completed!" << std::endl;

	boundary_particle = std::vector<Particle>();
	addBoundaryParticles(interval, supportRadius);
	//addObstacle(interval, supportRadius);
	addBunny("../bunny.obj", supportRadius);
	boundary_number = boundary_particle.size();
	std::cout << boundary_number << " boundary particles has been displaced!" << std::endl;
	block_storage = std::vector<Block>(block_number);
	block_storage = std::vector<Block>(block_number);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < boundary_number; i++)
	{
		Particle &particle = boundary_particle[i];
		particle.index = fluid_number+i;
		particle.assigned_to_block( block_storage, block_unit, blockSize, boundary );
	}
}



SPH::SPH(const int even[3], 
		 const float& time_step, 
		 const int& iteration, 
		 const float& interval)
{
	fluid_number = even[0]*even[1]*even[2];
	this->iteration = iteration;
	this->time_step = time_step;
	
	const float& supportRadius = 2.0*interval;
	blockSize = interval;
	const float& mass = 0.8*rest_density*pow(interval,3.0);
	const float external_foce[] = {0.0, 0.0, -gravity*mass};
	const float velo[3] = {0.0};

	std::cout << "Distance among particles are " << interval << " " << interval << " " << interval << std::endl;
	
	const float x[2] = {-supportRadius, interval*even[0]*(float)2.2};
	const float y[2] = {-supportRadius, interval*even[1]*(float)1.2};
	const float z[2] = {-supportRadius, interval*even[2]*(float)2.2};

	boundary = Grid(x, y, z);

	std::cout << "Boudnary domain is: [" << boundary.x_range.x << ", " << boundary.x_range.y << "] X [" 
										 << boundary.y_range.x << ", " << boundary.y_range.y << "] X ["
										 << boundary.z_range.x << ", " << boundary.z_range.y << "]!" << std::endl;

	block_unit[0] = ceil((boundary.x_range.y - boundary.x_range.x)/blockSize) + 1;
	block_unit[1] = ceil((boundary.y_range.y - boundary.y_range.x)/blockSize) + 1;
	block_unit[2] = ceil((boundary.z_range.y - boundary.z_range.x)/blockSize) + 1; // compute the block unit
	block_number = block_unit[0]*block_unit[1]*block_unit[2];

	particle_list = std::vector<Particle>(fluid_number);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i = 0; i < fluid_number; i++)
	{
		float posi[3];
		posi[0] = (i%even[0]) * interval + boundary.x_range.x + interval/*supportRadius*/;
		posi[1] = (i/even[0]%even[1]) * interval + boundary.y_range.x +interval/*supportRadius*/;
	    posi[2] = (i/even[0]/even[1]) * interval + boundary.z_range.x +interval/*supportRadius*/;
		const float velo[3] = {(float)0};
		particle_list[i] = Particle(mass, posi, velo, supportRadius, external_foce, i);
	}

	boundary_particle = std::vector<Particle>();
	addBoundaryParticles(interval, supportRadius);
	//addObstacle(interval, supportRadius);
	addBunny("../bunny.obj", supportRadius);
	boundary_number = boundary_particle.size();
	std::cout << boundary_number << " boundary particles has been displaced!" << std::endl;
	block_storage = std::vector<Block>(block_number);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < boundary_number; i++)
	{
		Particle &particle = boundary_particle[i];
		particle.index = fluid_number+i;
		particle.assigned_to_block( block_storage, block_unit, blockSize, boundary);
	}
	printBoundary();
	std::cout << fluid_number << " fluid particles distribution completed!" << std::endl;
}



void SPH::addWall(const float min_[3], 
				  const float max_[3], 
				  const float& interval, 
				  const float& radius, 
				  float *normal)
{
	const int& stepsX = (max_[0] - min_[0])/interval + 1; 
	const int& stepsY = (max_[1] - min_[1])/interval + 1;
	const int& stepsZ = (max_[2] - min_[2])/interval + 1;
	const int& start = boundary_particle.size();
	boundary_particle.resize(start + stepsX*stepsY*stepsZ);

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int j = 0; j < (int)stepsX; j++)
	{
		for (unsigned int k = 0; k < stepsY; k++)
		{
			for (unsigned int l = 0; l < stepsZ; l++)
			{
				const float currPos[3] = {min_[0]+j*interval, min_[1]+k*interval, min_[2]+l*interval};
				boundary_particle[start + j*stepsY*stepsZ + k*stepsZ + l] = Particle(currPos, radius, normal);
			}
		}
	}
}

void SPH::addObstacle(const float& interval, 
					  const float& supportRadius)
{
	const float x_[2] = {80,100};
	const float y_[2] = {5,50};
	const float z_[2] = {-1.8,40};

	obstacle = Grid(x_, y_, z_);

	const float vertex1[] = {obstacle.x_range.x, obstacle.y_range.x, obstacle.z_range.x};
	const float vertex2[] = {obstacle.x_range.x, obstacle.y_range.x, obstacle.z_range.y};
	const float vertex3[] = {obstacle.x_range.x, obstacle.y_range.y, obstacle.z_range.x};
	const float vertex4[] = {obstacle.x_range.x, obstacle.y_range.y, obstacle.z_range.y};
	const float vertex5[] = {obstacle.x_range.y, obstacle.y_range.x, obstacle.z_range.x};
	const float vertex6[] = {obstacle.x_range.y, obstacle.y_range.x, obstacle.z_range.y};
	const float vertex7[] = {obstacle.x_range.y, obstacle.y_range.y, obstacle.z_range.x};
	const float vertex8[] = {obstacle.x_range.y, obstacle.y_range.y, obstacle.z_range.y};

	float normal_[6][3] = { 0,0,1,
							1,0,0,
							0,0,-1,
							-1,0,0,
							0,-1,0,
							0,1,0};

	addWall(vertex2, vertex8, interval, supportRadius, &(normal_[0][0]));
	addWall(vertex5, vertex8, interval, supportRadius, &(normal_[1][0]));
	addWall(vertex1, vertex7, interval, supportRadius, &(normal_[2][0]));
	addWall(vertex1, vertex4, interval, supportRadius, &(normal_[3][0]));
	addWall(vertex1, vertex6, interval, supportRadius, &(normal_[4][0]));
	addWall(vertex3, vertex8, interval, supportRadius, &(normal_[5][0]));

	printObstacle(obstacle, 0);
}


void SPH::addBoundaryParticles(const float& interval, 
							   const float& supportRadius)
{
	const float vertex1[] = {boundary.x_range.x, boundary.y_range.x, boundary.z_range.x};
	const float vertex2[] = {boundary.x_range.x, boundary.y_range.x, boundary.z_range.y};
	const float vertex3[] = {boundary.x_range.x, boundary.y_range.y, boundary.z_range.x};
	const float vertex4[] = {boundary.x_range.x, boundary.y_range.y, boundary.z_range.y};
	const float vertex5[] = {boundary.x_range.y, boundary.y_range.x, boundary.z_range.x};
	const float vertex6[] = {boundary.x_range.y, boundary.y_range.x, boundary.z_range.y};
	const float vertex7[] = {boundary.x_range.y, boundary.y_range.y, boundary.z_range.x};
	const float vertex8[] = {boundary.x_range.y, boundary.y_range.y, boundary.z_range.y};
	
	float normal_[6][3] = { 0,0,-1,
							-1,0,0,
							0,0,1,
							1,0,0,
							0,1,0,
							0,-1,0};

	addWall(vertex2, vertex8, interval, supportRadius, &(normal_[0][0]));
	addWall(vertex5, vertex8, interval, supportRadius, &(normal_[1][0]));
	addWall(vertex1, vertex7, interval, supportRadius, &(normal_[2][0]));
	addWall(vertex1, vertex4, interval, supportRadius, &(normal_[3][0]));
	addWall(vertex1, vertex6, interval, supportRadius, &(normal_[4][0]));
	addWall(vertex3, vertex8, interval, supportRadius, &(normal_[5][0]));

}


void SPH::addBunny(const char* fileName, 
				   const float& radius)
{
	ifstream fin(fileName, ios::in);
	if(!fin)
	{
		cout << "Error creating a file!" << endl;
		exit(-1);
	}

	string iline;
	for (int i = 0; i < 8; ++i)
	{
		getline(fin, iline);
	}

	const int& result = iline.find("Vertices: ");
	const int& COUNT = atoi(iline.substr(result+10).c_str());
	cout << "Boundary " << fileName << " has " << COUNT << " boundary particles!" << endl;

	for (int i = 0; i < 3; ++i)
	{
		getline(fin, iline);
	}

	stringstream ss;
	string vn;
	float value[3], normal[3], normSum;

	const int& start = boundary_particle.size();
	std::cout << start << std::endl;
	boundary_particle.resize(start + COUNT);

	for (int i = 0; i < COUNT; ++i)
	{
		getline(fin, iline);
		ss << iline;
		ss >> vn >> normal[0] >> normal[1] >> normal[2];
		normSum = 0.0;
		for (int j = 0; j < 3; ++j)
		{
			normSum += normal[j]*normal[j];
		}
		normSum = sqrt(normSum);
		for (int j = 0; j < 3; ++j)
		{
			normal[j]/=normSum;
		}
		ss.clear();
		ss.str("");
		getline(fin, iline);
		ss << iline;
		ss >> vn >> value[0] >> value[1] >> value[2];
		boundary_particle[start + i] = Particle(value, radius, normal);
		ss.clear();
		ss.str("");
	}

	fin.close();
}



void SPH::addWaterVolume(const float& mass, 
						 const float& supportRadius, 
						 const float external_force[], 
						 const float& interval, 
						 const int even[])
{
	particle_list.resize(fluid_number*2);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i = 0; i < fluid_number; i++)
	{
		float posi[3];
		posi[0] = (i%even[0]) * interval + boundary.x_range.x + interval/*supportRadius*/;
		posi[1] = (i/even[0]%even[1]) * interval + boundary.y_range.x + interval/*supportRadius*/;
	    posi[2] = (i/even[0]/even[1]) * interval + boundary.z_range.y/2.0 + interval/*supportRadius*/;
		const float velo[3] = {(float)0};
		particle_list[i + fluid_number] = Particle(mass, posi, velo, supportRadius, external_force, i + fluid_number);
	}
	fluid_number = particle_list.size();
}



void SPH::addBunnyWater(const string& fileName, 
						const float& mass, 
						const float& supportRadius, 
						const float& interval,
						const float external_force[])
{
	std::vector<std::vector< float> > vertices;
	std::vector<std::vector< int> > triangles;
	float posLimit[][2] = {FLT_MAX, FLT_MIN,
					 	   FLT_MAX, FLT_MIN,
				    	   FLT_MAX, FLT_MIN};

	getVertexFace(fileName, vertices, triangles, posLimit);

	for (int i = 0; i < 3; ++i)
	{
		std::cout << posLimit[i][0] << " " << posLimit[i][1] << std::endl;
	}

	const int& Vertex = vertices.size();
	const int& Faces = triangles.size();

	const int xStep = (posLimit[0][1]-posLimit[0][0])/interval + 1;
	const int yStep = (posLimit[1][1]-posLimit[1][0])/interval + 1;
	const int zStep = (posLimit[2][1]-posLimit[2][0])/interval + 1;

	float xPos, yPos, zPos;
	float pos[3], velo[3]= {0.0};
	for (int i=0; i<xStep; i++)
	{
		for (int j = 0; j < yStep; ++j)
		{
			for (int k = 0; k < zStep; ++k)
			{
				xPos = posLimit[0][0]+interval*i;
				yPos = posLimit[1][0]+interval*j;
				zPos = posLimit[2][0]+interval*k;
				if(stayInVolume(xPos,yPos,zPos,vertices, triangles))
				{
					pos[0] = xPos, pos[1] = yPos, pos[2] = zPos;
					particle_list.push_back(Particle(mass, pos, velo, supportRadius, external_force, fluid_number++));
				}
			}
		}
	}
	fluid_number = particle_list.size();
	particle_list.resize(fluid_number+Vertex);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i = 0; i < Vertex; i++)
	{
		float posi[3] = {vertices[i][0], vertices[i][1], vertices[i][2]};
		const float velo[3] = {(float)0};
		particle_list[i + fluid_number] = Particle(mass, posi, velo, supportRadius, external_force, i + fluid_number);
	}
	fluid_number = particle_list.size();
}



bool SPH::stayInVolume(const float& x, 
					   const float& y, 
					   const float& z, 
					   const std::vector<std::vector<float> >& vertices, 
					   const std::vector<std::vector<int> >& triangles) const
{
/* Use ray casting method to sample points inside complicated geometric volume,
   we use x >=x0, y = y0, z = z0 this ray to travel through the volume.
   If intersection point number is odd, then the point (x0,y0,z0) lines inside,
   otherwise, it stays outside the volume */
	std::vector<float> point0, point1, point2;
	float yMin, yMax, xMin, xMax, xSolution;
	glm::vec3 v0, v1, v2, orig, position;
	glm::vec3 direction(0,0,1);
	bool inside = false;
	orig = glm::vec3(x,y,z);
	for (int i = 0; i < triangles.size(); ++i)
	{
		point0 = vertices[triangles[i][0]];
		point1 = vertices[triangles[i][1]];
		point2 = vertices[triangles[i][2]];
		yMin = std::min(std::min(point0[1], point1[1]), point2[1]);
		yMax = std::max(std::max(point0[1], point1[1]), point2[1]);
		xMin = std::min(std::min(point0[0], point1[0]), point2[0]);
		xMax = std::max(std::max(point0[0], point1[0]), point2[0]);
		if(y>yMin&&y<yMax&&x>xMin&&x<xMax)
		{
			v0 = glm::vec3(point0[0], point0[1], point0[2]);
			v1 = glm::vec3(point1[0], point1[1], point1[2]);
			v2 = glm::vec3(point2[0], point2[1], point2[2]);
			if(glm::intersectRayTriangle(orig, direction, v0, v1, v2, position)
			   ||glm::intersectRayTriangle(orig, direction, v0, v2, v1, position))
        	{
            	inside = !inside;
        	}
		}
	}
	return inside;
}



void SPH::getVertexFace(const string& fileName, 
					    std::vector<std::vector<float> >& vertices, 
				  	 	std::vector<std::vector<int> >& triangles, 
				  	 	float limit[3][2])
{
	ifstream fin(fileName.c_str(), ios::in);
	if(!fin)
	{
		std::cout << "Error reading file!" << std::endl;
		exit(1);
	}
	size_t vertexCount, meshCount;
	string line;
	for (int i = 0; i < 8; ++i)
	{
		getline(fin,line);
	}
	vertexCount = line.find("Vertices: ");
	vertexCount = atoi(line.substr(vertexCount+10).c_str());
	getline(fin,line);
	meshCount = line.find("Faces: ");
	meshCount = atoi(line.substr(meshCount+6).c_str());
	std::cout << vertexCount << " vertices and " << meshCount << " triangles!" << std::endl;

	vertices = std::vector<std::vector<float> >(vertexCount, std::vector<float>(3));
	triangles = std::vector<std::vector<int> >(meshCount, std::vector<int>(3));

	for (int i = 0; i < 2; ++i)
	{
		getline(fin,line);
	}

	float pos[3];
	string vStr;
	stringstream ss;
	for (int i = 0; i < vertexCount; ++i)
	{
		getline(fin,line);
		getline(fin,line);
		ss << line;
		ss >> vStr >> pos[0] >> pos[1] >> pos[2];
		for (int j = 0; j < 3; ++j)
		{
			if(limit[j][0]>pos[j])
				limit[j][0]=pos[j];
			if(limit[j][1]<pos[j])
				limit[j][1]=pos[j];
			vertices[i][j] = pos[j];
		}
		ss.clear();
		ss.str("");
	}
	for (int i = 0; i < 2; ++i)
	{
		getline(fin,line);
	}

	int slashPos;
	for (int i = 0; i < meshCount; ++i)
	{
		getline(fin,line);
		ss << line;
		ss >> vStr;
		for (int j = 0; j < 3; ++j)
		{
			ss >> vStr;
			slashPos = vStr.find("//");
			triangles[i][j] = atoi(vStr.substr(0,slashPos).c_str());
		}
		ss.clear();
		ss.str("");
	}
	fin.close();
}



void SPH::addObstacle()
{
	const float x[] = {0,40};
	const float y[] = {-2,60};
	const float z[] = {85,120};
	obstacle = Grid(x, y, z);
	printObstacle(obstacle, 0);
}


void SPH::configure()
{
	block_storage.clear();
	block_storage = std::vector<Block>(block_number);

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < boundary_number; i++)
	{
		Particle &particle = boundary_particle[i];
		particle.assigned_to_block( block_storage, block_unit, blockSize, boundary );
	}

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < fluid_number; i++)
	{
		Particle& particle = particle_list[i];
		particle.assigned_to_block(block_storage, block_unit, blockSize, boundary);
		particle.neighbor.clear();
	}
}


void SPH::update_position_by_force()
{
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i = 0; i < fluid_number; i++)
	{
		Particle& particle = particle_list[i];
		const glm::vec3& acce_force = particle.external_foce/particle.mass;
		particle.velocity += time_step * acce_force;
		particle.position += time_step * particle.velocity;

		particle.boundaryClamp(boundary);
	}
}


void SPH::compute_boundary_neighbor()
{
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for( int i = 0; i < boundary_number; i++)
	{
		Particle& particle = boundary_particle[i];
		int distance_times[3];
		distance_times[0] = (int)ceil(particle.radius/blockSize);
		distance_times[1] = (int)ceil(particle.radius/blockSize);
		distance_times[2] = (int)ceil(particle.radius/blockSize);	
		const std::vector< int >& rough_series = find_block_by_layer( particle.posi_index, distance_times );

		const int& rough_length = rough_series.size();
		for ( int count = 0; count < rough_length; count++)
		{
			const Block& block = block_storage[rough_series[count]];
			if( block.size != 0)
			{
				const std::vector<int >& block_particle_list = std::vector<int>(block.inside, block.inside+block.size);
				for ( unsigned int inside_count = 0; inside_count < block_particle_list.size(); ++ inside_count )
				{
					int inside_index = block_particle_list[ inside_count ];
					if( inside_index == particle.index )
						continue;
					else
					{					
						glm::vec3 distance_radius;
						if (inside_index < fluid_number)
						{
							distance_radius = particle.position - particle_list[inside_index].position;
						}
						else
							distance_radius = particle.position - boundary_particle[inside_index - fluid_number].position;
						if ( glm::length( distance_radius) <= particle.radius )
							particle.neighbor.push_back( inside_index );
					}
				}
			}			
		}
		particle.num_neigh = particle.neighbor.size();	
	}
}


void SPH::computePsi()
{
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for(int i = 0; i < boundary_number; i++)
	{
		Particle& particle = boundary_particle[i];
		float temp = 0.0;
		for(int j = 0; j < particle.num_neigh; j++)
		{
			temp += Kernel::W_poly6(particle.position - boundary_particle[particle.neighbor[j]-fluid_number].position, particle.radius);
		}
		particle.mass = rest_density/temp;
	}
}



void SPH::compute_neighbor()
{
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for( int i = 0; i < fluid_number; i++)
	{
		Particle& particle = particle_list[i];
		int distance_times[3];
		distance_times[0] = (int)ceil(particle.radius/blockSize);
		distance_times[1] = (int)ceil(particle.radius/blockSize);
		distance_times[2] = (int)ceil(particle.radius/blockSize);	
		const std::vector< int >& rough_series = find_block_by_layer( particle.posi_index, distance_times );

		int rough_length = rough_series.size();
		for ( int count = 0; count < rough_length; count++)
		{
			const Block& block = block_storage[rough_series[count]];
			if( block.size != 0)
			{
				std::vector<int > block_particle_list = std::vector<int>(block.inside, block.inside+block.size);
				for ( unsigned int inside_count = 0; inside_count < block_particle_list.size(); ++ inside_count )
				{
					int inside_index = block_particle_list[ inside_count ];
					if( inside_index == particle.index )
						continue;
					else
					{					
						glm::vec3 distance_radius;
						if (inside_index < fluid_number)
						{
							distance_radius = particle.position - particle_list[inside_index].position;
						}
						else
							distance_radius = particle.position - boundary_particle[inside_index - fluid_number].position;
						if ( glm::length( distance_radius) <= particle.radius )
							particle.neighbor.push_back( inside_index );
					}
				}
			}			
		}
		particle.num_neigh = particle.neighbor.size();
	}
}



void SPH::update_position_by_iteration( const float& factor_epsilon, 
										const int& artificialPressure)
{
	int current_iter = 0;
	while ( current_iter < iteration)
	{
	#pragma omp parallel for schedule(dynamic) num_threads(8)	
		for ( int i = 0; i < fluid_number; ++ i)
		{
			Particle& particle = particle_list[i];
			particle.compute_factor(factor_epsilon, particle_list, boundary_particle);
		}

	#pragma omp parallel for schedule(dynamic) num_threads(8)	
		for ( int particle_count = 0; particle_count < fluid_number; ++ particle_count)
		{
			Particle& particle = particle_list[particle_count];
			particle.compute_position_update_by_factor(artificialPressure, particle_list, boundary_particle);			
		}

	#pragma omp parallel for schedule(dynamic) num_threads(8)	
		for ( int update = 0; update < fluid_number; ++ update )
		{
			Particle& particle = particle_list[update];
			particle.position += particle.delta_position;
			particle.boundaryClamp(boundary);
		}
		current_iter++;
	}
}


void SPH::update_position_by_vorticity_and_XSPH(const float& vorticity_epsilon, 
												const int& vorticityConfinement,
												const int& surfaceOption)
{
#pragma omp parallel for schedule(dynamic) num_threads(8)	
	for ( int j = 0; j < fluid_number; j++)
	{
		Particle& particle = particle_list[j];
		particle.velocity = (particle.position - particle.original_position)/ time_step;
	}

#pragma omp parallel for schedule(dynamic) num_threads(8)	
	for ( int j = 0; j < fluid_number; j++)
	{
		Particle& particle = particle_list[j];
		particle.compute_vorticity(particle_list, boundary_particle, time_step);
	}

	if(vorticityConfinement)
	{
	#pragma omp parallel for schedule(dynamic) num_threads(8)	
		for ( int j = 0; j < fluid_number; j++)
		{
			Particle& particle = particle_list[j];
			particle.get_velocity_by_vorticity(vorticity_epsilon, particle_list, 
											   boundary_particle, time_step);
		}
	}

	switch(surfaceOption)
	{
	case 1:
		SurfaceTension::addWCSPH_surfaceTension(boundary_particle, particle_list, time_step);
		break;

	case 2:
		SurfaceTension::addVersatle_surfaceTension(boundary_particle, particle_list, time_step);
		break;

	case 3:
		SurfaceTension::addHe_surfaceTension(boundary_particle, particle_list, time_step);
		break;

	default:
		SurfaceTension::noTension(boundary_particle, particle_list, time_step);
		break;
	}
}


void SPH::printBoundaryParticlesVTK()
{
	ofstream fout("../boundaryParticles.vtk", ios::out);
	fout << "# vtk DataFile Version 3.0" << endl;
	fout << "PBF example" << endl;
	fout << "ASCII" << endl;
	fout << "DATASET UNSTRUCTURED_GRID" << endl;
	fout << "POINTS " << boundary_number << " float" << endl;
	for( int i = 0; i < boundary_number; ++i)
	{
		const Particle& particle = boundary_particle[i];
		fout << particle.position.x << " " << particle.position.y << " " << particle.position.z << endl; 	 
	}
	fout << "CELLS " << boundary_number << " " << 2*boundary_number << endl;
	for (int i = 0; i < boundary_number; i++)
	{
		fout << 1 << " " << i << endl;
	}
	fout << "CELL_TYPES " << boundary_number << endl;
	for (int i = 0; i < boundary_number; i++)
	{
		fout << 1 << endl;
	}
	fout << "POINT_DATA " << boundary_number << endl;
	fout << "SCALARS mass float 1" << endl;
	fout << "LOOKUP_TABLE mass_table" << endl;
	for ( int i = 0; i < boundary_number; i++)
	{
		fout << boundary_particle[i].mass << endl;
	}
	fout.close();
}

void SPH::output(const int& frame, 
				 float extrema[6])
{

/*-------------------- Output vtk format for Paraview to visualize the fluid simulation data ---------------------------------------------------*/
	std::ofstream fout;
	string frame_str;
	stringstream ss;
	ss << frame;
	ss >> frame_str;
	fout.open( ("../sourceData/frame" + frame_str + ".vtk").c_str(), ios::out);
	fout << "# vtk DataFile Version 3.0" << endl;
	fout << "PBF example" << endl;
	fout << "ASCII" << endl;
	fout << "DATASET UNSTRUCTURED_GRID" << endl;
	fout << "POINTS " << fluid_number << " float" << endl;
	for( int i = 0; i < fluid_number; ++i)
	{
		const Particle& particle = particle_list[i];
		fout << particle.position.x << " " << particle.position.y << " " << particle.position.z << endl; 	 
	}
	fout << "CELLS " << fluid_number << " " << 2*fluid_number << endl;
	for (int i = 0; i < fluid_number; i++)
	{
		fout << 1 << " " << i << endl;
	}
	fout << "CELL_TYPES " << fluid_number << endl;
	for (int i = 0; i < fluid_number; i++)
	{
		fout << 1 << endl;
	}
	fout << "POINT_DATA " << fluid_number << endl;
	fout << "SCALARS velocity_norm float 1" << endl;
	fout << "LOOKUP_TABLE velocity_table" << endl;
	for ( int i = 0; i < fluid_number; i++)
	{
		float velocity = glm::length(particle_list[i].velocity);
		fout << velocity << endl;
		if (velocity < extrema[0])
		{
			extrema[0] = velocity;
		}
		if (velocity > extrema[1])
		{
			extrema[1] = velocity;
		}
	}
	fout << "SCALARS Vorticity float 1" << endl;
	fout << "LOOKUP_TABLE vorticity_table" << endl;
	for ( int i = 0; i < fluid_number; i++)
	{
		float vorticity = glm::length(particle_list[i].vorticity);
		fout << vorticity << endl;
		if (vorticity < extrema[4])
		{
			extrema[4] = vorticity;
		}
		if (vorticity > extrema[5])
		{
			extrema[5] = vorticity;
		}
	}
	fout << "SCALARS Density float 1" << endl;
	fout << "LOOKUP_TABLE density_table" << endl;
	for ( int i = 0; i < fluid_number; i++)
	{
		float density = particle_list[i].density;
		fout << density << endl;
		if (density < extrema[2])
		{
			extrema[2] = density;
		}
		if (density > extrema[3])
		{
			extrema[3] = density;
		}
	}
	fout.close();

	std::ofstream position_;

	position_.open(string("../sourceData/Frame " + frame_str + ".txt").c_str(), ios::out);
	for (int i = 0; i < fluid_number; i++)
	{
		const Particle& particle = particle_list[i];
		position_ << i << " " << particle.position.x << " " << particle.position.y << " " << particle.position.z << " " << 
		particle.velocity.x << " " << particle.velocity.y << " " << particle.velocity.z << " " << particle.density << endl;
	}
	position_.close();

}


const std::vector<int> SPH::find_block_by_layer(const int index[3], 
												const int *layer)
{
	std::vector<int> result;
	int x_lower = max(index[0] - layer[0],0); 
	int x_upper = min(index[0] + layer[0] + 1, block_unit[0]);
	int y_lower = max(index[1] - layer[1],0); 
	int y_upper = min(index[1] + layer[1] + 1, block_unit[1]);
	int z_lower = max(index[2] - layer[2],0);
	int z_upper = min(index[2] + layer[2] + 1, block_unit[2]);
	for (int i = x_lower; i < x_upper; i++)
	{
		for (int j = y_lower; j < y_upper; j++)
		{
			for (int k = z_lower; k < z_upper; k++)
			{
				result.push_back(i + j*block_unit[0] + k*block_unit[0]*block_unit[1]);
			}
		}
	}
	return result;
}


void SPH::printBoundary()
{
	ofstream fout("../boundary.vtk", ios::out);
	if(!fout)
	{
		std::cout << "Error creating files!" << std::endl;
		exit(1);
	}
	fout << "# vtk DataFile Version 2.0" << std::endl
		 << "Cube example" << std::endl
		 << "ASCII" << std::endl
		 << "DATASET POLYDATA" << std::endl
		 << "POINTS 8 float" << std::endl;
	fout << boundary.x_range.x << " " << boundary.y_range.x << " " << boundary.z_range.y << std::endl
		 << boundary.x_range.y << " " << boundary.y_range.x << " " << boundary.z_range.y << std::endl
		 << boundary.x_range.y << " " << boundary.y_range.y << " " << boundary.z_range.y << std::endl
		 << boundary.x_range.x << " " << boundary.y_range.y << " " << boundary.z_range.y << std::endl
		 << boundary.x_range.x << " " << boundary.y_range.x << " " << boundary.z_range.x << std::endl
		 << boundary.x_range.y << " " << boundary.y_range.x << " " << boundary.z_range.x << std::endl
		 << boundary.x_range.y << " " << boundary.y_range.y << " " << boundary.z_range.x << std::endl
		 << boundary.x_range.x << " " << boundary.y_range.y << " " << boundary.z_range.x << std::endl;
    fout << "POLYGONS 6 30" << std::endl
		 << "4 0 1 2 3" << std::endl
		 << "4 0 1 5 4" << std::endl
		 << "4 4 5 6 7" << std::endl
		 << "4 3 2 6 7" << std::endl
		 << "4 1 2 6 5" << std::endl
	     << "4 0 3 7 4" << std::endl
		 << "CELL_DATA 6" << std::endl
		 << "NORMALS cell_normals float" << std::endl
		 << "0 0 -1" << std::endl
		 << "0 1 0" << std::endl
		 << "0 0 1" << std::endl
		 << "0 -1 0" << std::endl
		 << "-1 0 0" << std::endl
		 << "1 0 0" << std::endl;
	fout.close();
}


void SPH::printObstacle(const Grid& cube, 
						const int& order)
{
	stringstream ss;
	ss << "../obstacle_" << order << ".vtk";
	ofstream fout(ss.str().c_str(), ios::out);
	if(!fout)
	{
		std::cout << "Error creating files!" << std::endl;
		exit(1);
	}
	fout << "# vtk DataFile Version 2.0" << std::endl
		 << "Cube example" << std::endl
		 << "ASCII" << std::endl
		 << "DATASET POLYDATA" << std::endl
		 << "POINTS 8 float" << std::endl;
	fout << cube.x_range.x << " " << cube.y_range.x << " " << cube.z_range.y << std::endl
		 << cube.x_range.y << " " << cube.y_range.x << " " << cube.z_range.y << std::endl
		 << cube.x_range.y << " " << cube.y_range.y << " " << cube.z_range.y << std::endl
		 << cube.x_range.x << " " << cube.y_range.y << " " << cube.z_range.y << std::endl
		 << cube.x_range.x << " " << cube.y_range.x << " " << cube.z_range.x << std::endl
		 << cube.x_range.y << " " << cube.y_range.x << " " << cube.z_range.x << std::endl
		 << cube.x_range.y << " " << cube.y_range.y << " " << cube.z_range.x << std::endl
		 << cube.x_range.x << " " << cube.y_range.y << " " << cube.z_range.x << std::endl;
    fout << "POLYGONS 6 30" << std::endl
		 << "4 0 1 2 3" << std::endl
		 << "4 0 1 5 4" << std::endl
		 << "4 4 5 6 7" << std::endl
		 << "4 3 2 6 7" << std::endl
		 << "4 1 2 6 5" << std::endl
	     << "4 0 3 7 4" << std::endl
		 << "CELL_DATA 6" << std::endl
		 << "NORMALS cell_normals float" << std::endl
		 << "0 0 1" << std::endl
		 << "0 -1 0" << std::endl
		 << "0 0 -1" << std::endl
		 << "0 1 0" << std::endl
		 << "1 0 0" << std::endl
		 << "-1 0 0" << std::endl;
	fout.close();
}

