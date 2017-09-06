#include "SPH.h"
#include <cmath>
#include<fstream>
#include<iostream>
#include<sstream>
#include<string>
using namespace std;


SPH::SPH(void)
{

}


SPH::~SPH(void)
{
	cout << "SPH destructor invoked!" << endl;
}


SPH::SPH(const string& file_)
{
	const int even[3] = {40, 40, 80};
	fluid_number = 128000;
	iteration = 3;
	time_step = 0.004;
	const float& interval = 0.05;
	const float& supportRadius = 2.0*interval;
	blockSize = supportRadius;
	const float& mass = 0.8*rest_density*pow(interval,3.0);
	const float external_foce[] = {0.0, -gravity*mass, 0.0};
	const float velo[3] = {0.0};

	std::cout << "Distance among particles are " << interval << " " << interval << " " << interval << std::endl;
	
	const float x[2] = {-supportRadius, interval*even[0]*1.2};
	const float y[2] = {-supportRadius, interval*even[1]*3.0};
	const float z[2] = {-supportRadius, interval*even[2]*1.6};

	boundary = Grid(x, y, z);

	//addObstacle(interval, supportRadius);

	std::cout << "Boudnary domain is: [" << boundary.x_range.x << ", " << boundary.x_range.y << "] X [" 
										 << boundary.y_range.x << ", " << boundary.y_range.y << "] X ["
										 << boundary.z_range.x << ", " << boundary.z_range.y << "]!" << std::endl;

	block_unit[0] = ceil((boundary.x_range.y - boundary.x_range.x)/blockSize) + 1;
	block_unit[1] = ceil((boundary.y_range.y - boundary.y_range.x)/blockSize) + 1;
	block_unit[2] = ceil((boundary.z_range.y - boundary.z_range.x)/blockSize) + 1; // compute the block unit
	block_number = block_unit[0]*block_unit[1]*block_unit[2];
	boundary_particle = std::vector<Particle>();


	addBoundaryParticles(interval, supportRadius);
	addBunny("my_bunny.ply", supportRadius);
	//addBunnyCenter("my_bunny.ply", supportRadius);

	boundary_number = boundary_particle.size();
	block_storage = std::vector<Block>(block_number);


#pragma omp parallel for num_threads(8)
	for (int i = 0; i < boundary_number; i++)
	{
		Particle &particle = boundary_particle[i];
		particle.index = fluid_number+i;
		particle.assigned_to_block( block_storage, block_unit, blockSize, boundary );
	}

		particle_list = std::vector<Particle>(fluid_number);
		

		ifstream fin(file_.c_str(), ios::in);
		if (fin.fail())
		{
			cout << "Error creating the files!" << endl;
			exit(-1);
		}


		float a, b, c, d, e;
		for(int i = 0; i < fluid_number; i++)
		{
			float posi[3];
			float velo[3];
			fin >> a >> posi[0] >> posi[1] >> posi[2] >> velo[0] >> velo[1] >> velo[2];
			fin >> b >> c >> d >> e;
			particle_list[i] = Particle(mass, posi, velo, supportRadius, external_foce, i);
		}
		std::cout << "Fluid particles distribution completed!" << std::endl;
}



SPH::SPH(const int even[3], const int& number, const float& time_step, const int& iteration, const float& interval)
{
	fluid_number = number;
	this->iteration = iteration;
	this->time_step = time_step;
	
	const float& supportRadius = 2.0*interval;
	blockSize = supportRadius;
	const float& mass = 0.8*rest_density*pow(interval,3.0);
	const float external_foce[] = {0.0, -gravity*mass, 0.0};
	const float velo[3] = {0.0};

	std::cout << "Distance among particles are " << interval << " " << interval << " " << interval << std::endl;
	
	const float x[2] = {-supportRadius, interval*even[0]*1.2};
	const float y[2] = {-supportRadius, interval*even[1]*3.0};
	const float z[2] = {-supportRadius, interval*even[2]*1.6};

	boundary = Grid(x, y, z);

	std::cout << "Boudnary domain is: [" << boundary.x_range.x << ", " << boundary.x_range.y << "] X [" 
										 << boundary.y_range.x << ", " << boundary.y_range.y << "] X ["
										 << boundary.z_range.x << ", " << boundary.z_range.y << "]!" << std::endl;

	block_unit[0] = ceil((boundary.x_range.y - boundary.x_range.x)/blockSize) + 1;
	block_unit[1] = ceil((boundary.y_range.y - boundary.y_range.x)/blockSize) + 1;
	block_unit[2] = ceil((boundary.z_range.y - boundary.z_range.x)/blockSize) + 1; // compute the block unit
	block_number = block_unit[0]*block_unit[1]*block_unit[2];
	boundary_particle = std::vector<Particle>();

	addBoundaryParticles(interval, supportRadius);

	//addObstacle(interval, supportRadius);
	addBunny("my_bunny.ply", supportRadius);
	//addBunnyCenter("my_bunny.ply", supportRadius);

	boundary_number = boundary_particle.size();
	block_storage = std::vector<Block>(block_number);
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < boundary_number; i++)
	{
		Particle &particle = boundary_particle[i];
		particle.index = fluid_number+i;
		particle.assigned_to_block( block_storage, block_unit, blockSize, boundary );
	}

	particle_list = std::vector<Particle>(fluid_number);
#pragma omp parallel for num_threads(8)
	for(int i = 0; i < fluid_number; i++)
	{
		float posi[3];
		posi[0] = (i%even[0]) * interval + boundary.x_range.x + interval/*supportRadius*/;
		posi[1] = (i/even[0]%even[1]) * interval + boundary.y_range.x + interval/*supportRadius*/;
	    posi[2] = (i/even[0]/even[1]) * interval + boundary.z_range.x + interval/*supportRadius*/;
		const float velo[3] = {(float)0};
		particle_list[i] = Particle(mass, posi, velo, supportRadius, external_foce, i);
	}
	std::cout << "Fluid particles distribution completed!" << std::endl;
}

void SPH::addObstacle(const float& interval, const float& supportRadius)
{
	const float x_[2] = {0.1, 2.0};
	const float y_[2] = {-0.095, 2.0};
	const float z_[2] = {4.5, 5.5};

	obstacle = Grid(x_, y_, z_);

	const float vertex1_[] = {obstacle.x_range.x, obstacle.y_range.x, obstacle.z_range.x};
	const float vertex2_[] = {obstacle.x_range.x, obstacle.y_range.x, obstacle.z_range.y};
	const float vertex3_[] = {obstacle.x_range.x, obstacle.y_range.y, obstacle.z_range.x};
	const float vertex4_[] = {obstacle.x_range.x, obstacle.y_range.y, obstacle.z_range.y};
	const float vertex5_[] = {obstacle.x_range.y, obstacle.y_range.x, obstacle.z_range.x};
	const float vertex6_[] = {obstacle.x_range.y, obstacle.y_range.x, obstacle.z_range.y};
	const float vertex7_[] = {obstacle.x_range.y, obstacle.y_range.y, obstacle.z_range.x};
	const float vertex8_[] = {obstacle.x_range.y, obstacle.y_range.y, obstacle.z_range.y};

	addWall(vertex2_, vertex8_, interval, supportRadius);
	addWall(vertex5_, vertex8_, interval, supportRadius);
	addWall(vertex1_, vertex7_, interval, supportRadius);
	addWall(vertex1_, vertex4_, interval, supportRadius);
	addWall(vertex1_, vertex6_, interval, supportRadius);
	addWall(vertex3_, vertex8_, interval, supportRadius);
}



void SPH::addBoundaryParticles(const float& interval, const float& supportRadius)
{
	const float vertex1[] = {boundary.x_range.x, boundary.y_range.x, boundary.z_range.x};
	const float vertex2[] = {boundary.x_range.x, boundary.y_range.x, boundary.z_range.y};
	const float vertex3[] = {boundary.x_range.x, boundary.y_range.y, boundary.z_range.x};
	const float vertex4[] = {boundary.x_range.x, boundary.y_range.y, boundary.z_range.y};
	const float vertex5[] = {boundary.x_range.y, boundary.y_range.x, boundary.z_range.x};
	const float vertex6[] = {boundary.x_range.y, boundary.y_range.x, boundary.z_range.y};
	const float vertex7[] = {boundary.x_range.y, boundary.y_range.y, boundary.z_range.x};
	const float vertex8[] = {boundary.x_range.y, boundary.y_range.y, boundary.z_range.y};
	

	addWall(vertex2, vertex8, interval, supportRadius);
	addWall(vertex5, vertex8, interval, supportRadius);
	addWall(vertex1, vertex7, interval, supportRadius);
	addWall(vertex1, vertex4, interval, supportRadius);
	addWall(vertex1, vertex6, interval, supportRadius);
	addWall(vertex3, vertex8, interval, supportRadius);

}



void SPH::addBunny(const char* fileName, const float& radius)
{
	ifstream fin(fileName, ios::in);
	if(!fin)
	{
		cout << "Error creating a file!" << endl;
		exit(-1);
	}

	string iline;
	for (int i = 0; i < 4; ++i)
	{
		getline(fin, iline);
	}

	const int& result = iline.find("x");
	const int& COUNT = atoi(iline.substr(result+2).c_str());
	cout << "Boundary " << fileName << " has " << COUNT << " boundary particles!" << endl;

	for (int i = 0; i < 8; ++i)
	{
		getline(fin, iline);
	}

	stringstream ss;
	float a, b, value[3];

	const int& start = boundary_particle.size();
	boundary_particle.resize(start + COUNT);

	for (int i = 0; i < COUNT; ++i)
	{
		getline(fin, iline);
		ss << iline;
		ss >> value[0] >> value[1] >> value[2] >> a >> b;
		ss.clear();
		boundary_particle[start + i] = Particle(value, radius);
		ss.clear();
	}

	fin.close();
	boundary_number = boundary_particle.size();
}

void SPH::addBunnyCenter(const char* fileName, const float& radius)
{
	ifstream fin(fileName, ios::in);
	if(!fin)
	{
		cout << "Error creating a file!" << endl;
		exit(-1);
	}

	string iline;
	for (int i = 0; i < 4; ++i)
	{
		getline(fin, iline);
	}

	int result = iline.find("x");
	const int& COUNT = atoi(iline.substr(result+2).c_str());
	cout << "Boundary " << fileName << " has " << COUNT << " boundary particles!" << endl;

	for (int i = 0; i < 6; ++i)
	{
		getline(fin, iline);
	}

	result = iline.find("c");

	const int& FACE = atoi(iline.substr(result+3).c_str());

	cout << "Bunny triangles " << fileName << " has " << FACE << " boundary particles!" << endl;

	float **position_ = new float*[COUNT];
	for (int i = 0; i < COUNT; ++i)
	{
		position_[i] = new float[3];
	}

	for (int i = 0; i < 2; ++i)
	{
		getline(fin, iline);
	}

	stringstream ss;
	float a, b;

	const int& start = boundary_particle.size();
	boundary_particle.resize(start + FACE);

	for (int i = 0; i < COUNT; ++i)
	{
		getline(fin, iline);
		ss << iline;
		ss >> position_[i][0] >> position_[i][1] >> position_[i][2] >> a >> b;
		//boundary_particle[start + i] = Particle(value, radius);
		ss.clear();
	}

	int first, second, third;
	float value[3];
	for (int i = 0; i < FACE; ++i)
	{
		getline(fin, iline);
		ss << iline;
		ss >> a >> first >> second >> third;
		value[0] = (position_[first][0]+position_[second][0]+position_[third][0])/3.0;
		value[1] = (position_[first][1]+position_[second][1]+position_[third][1])/3.0;
		value[2] = (position_[first][2]+position_[second][2]+position_[third][2])/3.0;
		boundary_particle[start + i] = Particle(value, radius);
		ss.clear();
	}
	fin.close();

	for (int i = 0; i < COUNT; ++i)
	{
		delete[] position_[i];
	}
	delete[] position_;

	boundary_number = boundary_particle.size();
}




void SPH::addWall(const float min_[3], const float max_[3], const float& interval, const float& radius)
{
	const int& stepsX = (max_[0] - min_[0])/interval + 1; 
	const int& stepsY = (max_[1] - min_[1])/interval + 1;
	const int& stepsZ = (max_[2] - min_[2])/interval + 1;
	const int& start = boundary_particle.size();
	boundary_particle.resize(start + stepsX*stepsY*stepsZ);

#pragma omp parallel for num_threads(8)
	for (int j = 0; j < (int)stepsX; j++)
	{
		for (unsigned int k = 0; k < stepsY; k++)
		{
			for (unsigned int l = 0; l < stepsZ; l++)
			{
				const float currPos[3] = {min_[0]+j*interval, min_[1]+k*interval, min_[2]+l*interval};
				boundary_particle[start + j*stepsY*stepsZ + k*stepsZ + l] = Particle(currPos, radius);
			}
		}
	}
}


void SPH::configure(const int& frame, float extrema[6])
{
	block_storage.clear();
	block_storage = std::vector<Block>(block_number);

#pragma omp parallel for num_threads(8)
	for (int i = 0; i < boundary_number; i++)
	{
		Particle &particle = boundary_particle[i];
		particle.assigned_to_block( block_storage, block_unit, blockSize, boundary );
	}

#pragma omp parallel for num_threads(8)
	for (int i = 0; i < fluid_number; i++)
	{
		Particle& particle = particle_list[i];
		const bool& result = particle.assigned_to_block(block_storage, block_unit, blockSize, boundary);
		if (!result)
		{
			cout << "Index " << i << " has penetrated the boundary!" << endl;
			output(frame, extrema);
			exit(-1);
		}
		particle.neighbor.clear();
	}
}


void SPH::update_position_by_force()
{
#pragma omp parallel for num_threads(8)
	for(int i = 0; i < fluid_number; i++)
	{
		Particle& particle = particle_list[i];

		const glm::vec3& acce_force = particle.external_foce/particle.mass;

		particle.velocity += time_step * acce_force;

		particle.delta_position = time_step * particle.velocity;

		particle.position += particle.delta_position;

		particle.boundaryClamp(boundary);
	}
}


void SPH::compute_boundary_neighbor()
{
#pragma omp parallel for num_threads(8)
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
				for ( unsigned int inside_count = 0; inside_count < block_particle_list.size(); inside_count++ )
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
#pragma omp parallel for num_threads(8)
	for(int i = 0; i < boundary_number; i++)
	{
		Particle& particle = boundary_particle[i];
		if (particle.num_neigh == 0)
		{
			cout << "Error! Boundary particle has no neighbor!" << endl;
			exit(-1);
		}
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
#pragma omp parallel for num_threads(8)
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
				for ( unsigned int inside_count = 0; inside_count < block_particle_list.size(); inside_count++ )
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



void SPH::update_position_by_iteration( const float& factor_epsilon)
{
	int current_iter = 0;
	while ( current_iter < iteration)
	{
	#pragma omp parallel for num_threads(8)	
		for ( int i = 0; i < fluid_number; i++)
		{
			Particle& particle = particle_list[i];
			particle.get_density(particle_list, boundary_particle);
			particle.compute_factor(factor_epsilon, particle_list, boundary_particle);
		}

	#pragma omp parallel for num_threads(8)	
		for ( int particle_count = 0; particle_count < fluid_number; ++ particle_count)
		{
			Particle& particle = particle_list[particle_count];
			particle.compute_position_update_by_factor(particle_list, boundary_particle);			
		}

	#pragma omp parallel for num_threads(8)	
		for ( int update = 0; update < fluid_number; ++ update )
		{
			Particle& particle = particle_list[update];
			particle.position.x += particle.delta_position.x;
			particle.position.y += particle.delta_position.y;
			particle.position.z += particle.delta_position.z;
			particle.boundaryClamp(boundary);
		}
		current_iter++;
	}
}


void SPH::update_position_by_vorticity_and_XSPH(const float& vorticity_epsilon) // a problem occurs whether vorticity force is applied the current time_step or next time_step
{
#pragma omp parallel for num_threads(8)	
	for ( int j = 0; j < fluid_number; j++)
	{
		Particle& particle = particle_list[j];
		particle.velocity.x = (particle.position.x - particle.original_position.x)/ time_step;
		particle.velocity.y = (particle.position.y - particle.original_position.y)/ time_step;
		particle.velocity.z = (particle.position.z - particle.original_position.z)/ time_step;
	}

#pragma omp parallel for num_threads(8)	
	for ( int j = 0; j < fluid_number; j++)
	{
		Particle& particle = particle_list[j];
		particle.compute_vorticity(particle_list, boundary_particle);
	}

#pragma omp parallel for num_threads(8)	
	for ( int j = 0; j < fluid_number; j++)
	{
		Particle& particle = particle_list[j];
		particle.compute_velocity_by_viscosity(particle_list, boundary_particle);
	}


#pragma omp parallel for num_threads(8)	
	for ( int j = 0; j < fluid_number; j++)
	{
		Particle& particle = particle_list[j];
		particle.get_velocity_by_vorticity(vorticity_epsilon, particle_list, boundary_particle,time_step);
		particle.original_position.x = particle.position.x;
		particle.original_position.y = particle.position.y;
		particle.original_position.z = particle.position.z;
	}


/*#pragma omp parallel for num_threads(8)	
	for ( int j = 0; j < fluid_number; j++)
	{
		Particle& particle = particle_list[j];
		particle.velocity.x = (particle.position.x - particle.original_position.x)/ time_step;
		particle.velocity.y = (particle.position.y - particle.original_position.y)/ time_step;
		particle.velocity.z = (particle.position.z - particle.original_position.z)/ time_step;
	}
	
#pragma omp parallel for num_threads(8)	
	for ( int j = 0; j < fluid_number; j++)
	{
		Particle& particle = particle_list[j];
		particle.compute_vorticity(particle_list, boundary_particle);
		particle.get_velocity_by_vorticity(vorticity_epsilon, particle_list, boundary_particle, time_step);
		particle.compute_velocity_by_viscosity(particle_list, boundary_particle);
		particle.original_position.x = particle.position.x;
		particle.original_position.y = particle.position.y;
		particle.original_position.z = particle.position.z;
	}*/
}


void SPH::output(const int& frame, float extrema[6])
{

/*-------------------- Output vtk format for Paraview to visualize the fluid simulation data ---------------------------------------------------*/

	std::ofstream position_;
	string frame_str;
	stringstream ss;
	ss << frame;
	ss >> frame_str;
	position_.open(string("Frame " + frame_str + ".txt").c_str(), ios::out);
	for (int i = 0; i < fluid_number; i++)
	{
		const Particle& particle = particle_list[i];
		position_ << i << " " << particle.position.x << " " << particle.position.y << " " << particle.position.z << " " << 
		particle.velocity.x << " " << particle.velocity.y << " " << particle.velocity.z << " " << particle.vorticity.x 
		<< " " << particle.vorticity.y << " " << particle.vorticity.z << " " << particle.density << endl;
	}
	position_.close();

	std::ofstream fout;
	fout.open( ("frame" + frame_str + ".vtk").c_str(), ios::out);
	fout << "# vtk DataFile Version 3.0" << endl;
	fout << "PBF example" << endl;
	fout << "ASCII" << endl;
	fout << "DATASET UNSTRUCTURED_GRID" << endl;
	fout << "POINTS " << fluid_number << " float" << endl;
	for( int i = 0; i < fluid_number; i++)
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
	fout << "SCALARS Velocity float 1" << endl;
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
}


const std::vector<int> SPH::find_block_by_layer(const int index[3], const int *layer)
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

