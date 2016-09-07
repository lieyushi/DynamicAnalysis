#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <climits>
#include <vector>
#include <ctime>
#include <eigen3/Eigen/Dense>
#include "Block.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_access.hpp"
using namespace std;
using namespace Eigen;
const int FRAME = 189;
const int PARTICLE = 128000;
const double ERROR = 1.0e-8;

const int NUMBER = 100;
double **data_information[NUMBER]; //this will store information from files
vector< vector<int > > neighbor(PARTICLE); //this will store neighbor of the current frame
vector< Hashing::Block> block_list;
const int& T = 40;
const double& radius = 0.1;
Hashing::Parameter para;
const double& sigma = 0.1;

const double Boundary[3][2] = {-0.1,2.4,
							   -0.1,3.0,
							   -0.1,10.0};


const double& TIMES = 1.1;
const int& THRESHOLD = 15;


const double distance(double *a, double *b);
const double Gaussian_weight(double *position, double *x, const double& sigma);
const vector<int> find_neigbhor_volume(const int& frame, double *position, const double& radius);
const double norm(double *a);
void initialize_memory_file_reader();
void release_memory_file_reader();
void load_data(const int& start);
void find_neigbhor_index(const int& frame, const int& number, double& radius);
void find_neighbor(const int& frame, const double& radius);
void my_init();
void assigned_grid(const int& frame, const double& radius);
void compute_FTLE(const int& frame);
double extrema[8] = {INT_MAX, INT_MIN, INT_MAX, INT_MIN, INT_MAX, INT_MIN, INT_MAX, INT_MIN};
/* store min and max values respectively for, density, velocity, ftle, rotation  */
void generate_txt(const int& row, const int& frame, double extrema[]);
// record minimal and maximal ftle and rotation
void volume_rendering(const int& row, const int& frame,const double& radius);                                                                                                                                                                                                                               void generate_vtk_particles(const int& row, const int& frame);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
int main(int argc, char **argv)
{
	double io_time = 0;
	clock_t starting, ending;
	time(&starting);
	initialize_memory_file_reader();
	
	/* will perform 7 times of I-O operations */
	for (int i = 0; i < ceil((double)(FRAME-T)/(double)(NUMBER-T)); i++)
	{
		clock_t begin, end;
		time(&begin);
		/* loading 305 files every time from 1 + 300*i */
		//load_data(1 + (NUMBER - T)*i);
		
		load_data((NUMBER - T)*i);
		time(&end);
		io_time += (double)difftime(end, begin);

		for (int j = 0; j < NUMBER - T; j++)//compute at most NUMBER-T+1 FTLE values
		{
			const int& frame = j + i*(NUMBER - T); //frame will be from 1 -> 1000 - 5
			if (frame > FRAME - T)
			{
				break;
			}
			cout << "Frame " << frame << " computation for FTLE starts!" << endl;

			assigned_grid(j, radius);			
			find_neighbor(j, radius);
			compute_FTLE(j);

			generate_txt(frame, j, extrema);

			volume_rendering(frame, j, radius);
			block_list.clear();
			generate_vtk_particles(frame, j);
			
			cout << "Frame " << frame << " computation ends!" << endl;
		}
	}

	release_memory_file_reader();
	time(&ending);

	ofstream out("README", ios::out);
	out << "Total computational time is: " << double(difftime(ending, starting)) << " S!" << endl;
	out << "I-O time is: " << io_time << " S" << endl;
	out << "Density range is: " << extrema[0] << ", " << extrema[1] << endl;
	out << "Velocity value is: " << extrema[2] << ", " << extrema[3] << endl;
	out << "FTLE value range is: " << extrema[4] << ", " << extrema[5] << endl;
	out << "Rotation value range is: " << extrema[6] << ", " << extrema[7] << endl;
	out.close();
	return 0;
}

void generate_vtk_particles(const int& frame, const int& j)
{
			string frame_str;
			stringstream ss;
			ss << frame;
			ss >> frame_str;
			frame_str = string("FTLE particle ") + frame_str + string(".vtk");
			ss.clear();
			ofstream fout(frame_str.c_str(), ios::out);
			if (!fout)
			{
				cout << "Error creating the file!" << endl;
			}

			fout << "# vtk DataFile Version 3.0" << endl;
			fout << "PBF_FTLE example" << endl;
			fout << "ASCII" << endl;
			fout << "DATASET UNSTRUCTURED_GRID" << endl;
			fout << "POINTS " << PARTICLE << " float" << endl;
			for( int k = 0; k < PARTICLE; ++k)
			{
				fout << data_information[j][k][0] << " " << data_information[j][k][1] 
				     << " " << data_information[j][k][2] << endl; 	 
			}
			fout << "CELLS " << PARTICLE << " " << 2*PARTICLE << endl;
			for (int k = 0; k < PARTICLE; k++)
			{
				fout << 1 << " " << k << endl;
			}
			fout << "CELL_TYPES " << PARTICLE << endl;
			for (int k = 0; k < PARTICLE; k++)
			{
				fout << 1 << endl;
			}
			fout << "POINT_DATA " << PARTICLE << endl;
			fout << "SCALARS density float 1" << endl;
			fout << "LOOKUP_TABLE ftletable" << endl;
			for ( int k = 0; k < PARTICLE; k++)
			{
				fout << data_information[j][k][6] << endl;
			}	

			fout << "SCALARS velocity float 1" << endl;
			fout << "LOOKUP_TABLE velocity_table" << endl;
			for ( int k = 0; k < PARTICLE; k++)
			{
				double v[] = {data_information[j][k][3], data_information[j][k][4], data_information[j][k][5]};
				fout << norm(v) << endl;
			}	

			fout << "SCALARS ftle float 1" << endl;
			fout << "LOOKUP_TABLE ftle_table" << endl;
			for ( int k = 0; k < PARTICLE; k++)
			{
				fout << data_information[j][k][14] << endl;
			}

			fout << "SCALARS rotation float 1" << endl;
			fout << "LOOKUP_TABLE rotation_table" << endl;
			for ( int k = 0; k < PARTICLE; k++)
			{
				fout << data_information[j][k][15] << endl;
			}
			fout.close(); 
}



const double Gaussian_weight(double *position, double *x, const double& sigma)
{
	return exp(-pow(distance(position,x),2.0)/2.0/pow(sigma,2.0));
}


const double norm(double *a)
{
	return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

void volume_rendering(const int& j, const int& frame, const double& radius)
{
    const double& interval = radius/2.0;

/*distribution  order of particles on the surface, firstly x, then y, then z */
	const int& XSize = (Boundary[0][1]-Boundary[0][0])/interval + 1;
	const int& YSize = (Boundary[1][1]-Boundary[1][0])/interval + 1;
	const int& ZSize = (Boundary[2][1]-Boundary[2][0])/interval + 1;
	const int& total = XSize*YSize*ZSize;
	double **scalar = new double*[total];

#pragma omp parallel for num_threads(8)
	for(int i = 0; i < total; i++)
	{
		scalar[i] = new double[4];
		scalar[i][0] = scalar[i][1] = scalar[i][2] = scalar[i][3] = 0.0;
	}

//#pragma omp parallel for num_threads(8)
	for(int i = 0; i < total; i++)
	{
		double position[3] = {i%XSize*interval + para.RANGE[0][0],
									i/XSize%YSize*interval + para.RANGE[1][0],
									i/XSize/YSize*interval + para.RANGE[2][0]};
		const vector<int>& support = find_neigbhor_volume(frame, position, radius);
		if(support.size() == 0)
			continue;
		double dorminant = 0.0;
		for(int j = 0; j < support.size(); j++)
		{
			double x[3] = {data_information[frame][support[j]][0], 
								 data_information[frame][support[j]][1], 
								 data_information[frame][support[j]][2]};

			const double& gaussion = Gaussian_weight(position, x, sigma);
			dorminant += gaussion;
			scalar[i][0] += data_information[frame][support[j]][6] * gaussion;
			double v[3] = {data_information[frame][support[j]][8], data_information[frame][support[j]][9], data_information[frame][support[j]][10]};
			scalar[i][1] += norm(v) * gaussion;
			scalar[i][2] += data_information[frame][support[j]][14] * gaussion;
			scalar[i][3] += data_information[frame][support[j]][15] * gaussion;
		}
		if(dorminant == 0)
		{
			cout << "Dorminant found to be zero!" << endl;
			cout << dorminant << endl;
			exit(-1);
		}
		scalar[i][0] /= dorminant;
		scalar[i][1] /= dorminant;
		scalar[i][2] /= dorminant;
		scalar[i][3] /= dorminant;
	}

	string frame_str;
	stringstream ss;
	ss << j;
	ss >> frame_str;
	frame_str = string("FTLE volume ") + frame_str + string(".vtk");
	ss.clear();
	ofstream fout(frame_str.c_str(), ios::out);
	if (!fout)
	{
		cout << "Error creating the file!" << endl;
	}
	fout << "# vtk DataFile Version 3.0" << endl;
	fout << "Volume example" << endl;
	fout << "ASCII" << endl;
	fout << "DATASET STRUCTURED_POINTS" << endl;
	fout << "DIMENSIONS " << XSize << " " << YSize << " " << ZSize << endl;
	fout << "ASPECT_RATIO " << interval << " " << interval << " " << interval << " " << endl;
	fout << "ORIGIN " << para.RANGE[0][0] << " " << para.RANGE[1][0] << " " << para.RANGE[2][0] << endl;
	fout << "POINT_DATA " << total << endl;
	fout << "SCALARS density float 1" << endl;
	fout << "LOOKUP_TABLE density_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][0] << endl;
	}

	fout << "SCALARS velocity float 1" << endl;
	fout << "LOOKUP_TABLE velocity_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][1] << endl;
	}

	fout << "SCALARS ftle float 1" << endl;
	fout << "LOOKUP_TABLE ftle_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][2] << endl;
	}

	fout << "SCALARS rotation float 1" << endl;
	fout << "LOOKUP_TABLE rotation_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][3] << endl;
	}
	fout.close();

	for(int i = 0; i < total; i++)
	{
		delete[] scalar[i];
	}
	delete[] scalar;
}


void generate_txt(const int& j, const int& frame, double extrema[])
{
	string frame_str;
	stringstream ss;
	ss << j;
	ss >> frame_str;
	frame_str = string("FTLE frame ") + frame_str + string(".txt");
	ss.clear();
	ofstream fout(frame_str.c_str(), ios::out);
	if (!fout)
	{
		cout << "Error creating the file!" << endl;
		exit(-1);
	}

	fout << "It has 12,8000 particles, data format is, position (3), velocity (3), density(1), neighbor_number(1), velocity_norm_gradient(3), " << 
			"density_gradient(3), ftle(1), rotation(1), velocity_Jacobian(9)!" << endl;
	for(int i = 0; i < PARTICLE; i++)
	{
		for(int j = 0; j < 25; j++)
		{
			fout << data_information[frame][i][j] << " ";
		}
		fout << endl;

		if(data_information[frame][i][6] < extrema[0])
			extrema[0] = data_information[frame][i][6];
		if(data_information[frame][i][6] > extrema[1])
			extrema[1] = data_information[frame][i][6];

		double v[3] = {data_information[frame][i][3], data_information[frame][i][4], data_information[frame][i][5]};
		if(norm(v) < extrema[2])
			extrema[2] = norm(v);
		if(norm(v) > extrema[3])
			extrema[3] = norm(v);	
		
		if(data_information[frame][i][14] < extrema[4])
			extrema[4] = data_information[frame][i][14];
		if(data_information[frame][i][14] > extrema[5])
			extrema[5] = data_information[frame][i][14];

		if(data_information[frame][i][15] < extrema[6])
			extrema[6] = data_information[frame][i][15];
		if(data_information[frame][i][15] > extrema[7])
			extrema[7] = data_information[frame][i][15];	
	}
	fout.close();
	cout << "Data written into text file for frame " << frame << endl;
}


void initialize_memory_file_reader()
/* memory allocation for data_information to store double[305][128000][4] */
{
#pragma omp parallel for num_threads(8)

	for(int i = 0; i < NUMBER; i++)
	{
		data_information[i] = new double *[PARTICLE];
		for (int j = 0; j < PARTICLE; j++)
		{
			data_information[i][j] = new double[25];
		}
	}
	std::cout << "Memory allocation completed!" << std::endl;
	std::cout << std::endl;
}



void release_memory_file_reader()
{
#pragma omp parallel for num_threads(8)

	for (int i = 0; i < NUMBER; i++)
	{
		for (int j = 0; j < PARTICLE; j++)
		{
			delete[] data_information[i][j];
		}
		delete[] data_information[i];
	}

#pragma omp parallel for num_threads(8)
	for (int i = 0; i < PARTICLE; i++)
	{
		neighbor[i].clear();
	}

	neighbor.clear();
	block_list.clear();
	std::cout << "Memory elimination completed! " << std::endl;
	std::cout << std::endl;
}


void load_data(const int& start) //start from 1
{
	const string& location = "/home/lieyu/Downloads/experiement_data/PBF_strictly_obeying_thePaper_details/";


#pragma omp parallel for num_threads(8)
	for (int i = start; i < (start + NUMBER <= (FRAME + 1)? start + NUMBER: FRAME + 1); i++)//loading files from 1->305, 301->605,...
	{
		string file_;
		stringstream ss;
		ss << i;
		ss >> file_;		
		file_ = location + string("Frame ") + file_ + string(".txt");
		ifstream in_(file_.c_str(), ios::in);
		if (!in_)
		{
			cout << "File doesn't exist in designated directory!" << endl;
			exit(1);
		}
		/* only load three elements of each line */
		double a,b;
		for (int j = 0; j < PARTICLE; j++)
		{
			in_ >> a;
			in_ >> data_information[i-start][j][0] >> data_information[i-start][j][1] >> data_information[i-start][j][2]
				>> data_information[i-start][j][3] >> data_information[i-start][j][4] >> data_information[i-start][j][5]
				>> data_information[i-start][j][6];
			in_ >> b;
		}
		ss.clear();
		in_.close();
		cout << "File " << (i) << " has been loaded successfully!" << endl;
	}

	cout << "Finished I-O Operations!" << endl;
}


void assigned_grid(const int& frame, const double& radius)
{
	/*   compute the member variables for para (Hashing::Parameter)   */
	para.compute_grid(Boundary, radius);
	cout << "Range is: [ " << para.RANGE[0][0] << ", " << para.RANGE[0][1] << "] X [ " <<
		para.RANGE[1][0] << ", " << para.RANGE[1][1] << "] X [ " << para.RANGE[2][0] << ", "
		<< para.RANGE[2][1] << "]." << endl;

	const int& b_size = para.grid[0]*para.grid[1]*para.grid[2];
	block_list = std::vector<Hashing::Block >(b_size);
	const int& size = block_list.size();
	
#pragma omp parallel for num_threads(8)
	/* assign each particle to its relative block */
	for (int i = 0; i < PARTICLE; i++)
	{
		const int& a = floor((data_information[frame][i][0] - para.RANGE[0][0])/para.grid_size[0]);
		const int& b = floor((data_information[frame][i][1] - para.RANGE[1][0])/para.grid_size[1]);
		const int& c = floor((data_information[frame][i][2] - para.RANGE[2][0])/para.grid_size[2]);
		const int& index = a + b*para.grid[0] + c*para.grid[0]*para.grid[1];
		assert(index >= 0 && index < size);
		block_list[index].add(i);
	}
	cout << "Frame " << frame << " assigning particle positions completed!" << endl;
}




void find_neighbor(const int& frame, const double& radius)
{
#pragma omp parallel for num_threads(8)
	/* firstly clear the former data in its neighbor */
	for (int i = 0; i < PARTICLE; i++)
	{
		neighbor[i].clear();
	}

	cout << "Starting searching neighbor!" << endl;
#pragma omp parallel for num_threads(8)	
	/* invoke neighbor information for each particle */
	for (int k = 0; k < PARTICLE; k++)
	{
		data_information[frame][k][7] = radius;
		find_neigbhor_index(frame, k, data_information[frame][k][7]);
		data_information[frame][k][7] = neighbor[k].size();
	}
}


const vector<int> find_neigbhor_volume(const int& frame, double *position, const double& radius)
{
		vector<int> my_neighbor;
		int ratio[3] = {ceil(radius/para.grid_size[0]),
						ceil(radius/para.grid_size[1]),
						ceil(radius/para.grid_size[2])};
		int index[3];

		index[0] = floor((position[0] - para.RANGE[0][0])/para.grid_size[0]);
		index[1] = floor((position[1] - para.RANGE[1][0])/para.grid_size[1]);
		index[2] = floor((position[2] - para.RANGE[2][0])/para.grid_size[2]);
		int x_lower = max(index[0] - ratio[0],0); 
		int x_upper = min(index[0] + ratio[0] + 1, para.grid[0]);
		int y_lower = max(index[1] - ratio[1],0); 
		int y_upper = min(index[1] + ratio[1] + 1, para.grid[1]);
		int z_lower = max(index[2] - ratio[2],0);
		int z_upper = min(index[2] + ratio[2] + 1, para.grid[2]);

		for (int i = x_lower; i < x_upper; i++)
		{
			for (int j = y_lower; j < y_upper; j++)
			{
				for (int k = z_lower; k < z_upper; k++)
				{
					const int& _num = para.grid[0]*para.grid[1]*k + para.grid[0]*j + i;
					if(!(_num >= 0 && _num < block_list.size()))
					{
						cout << _num << endl;
						cout << block_list.size() << endl;
						exit(-1);
					}
					//const vector<int>& inside = vector<int>(block_list[_num].inside, block_list[_num].inside+block_list[_num].size);
					if (block_list[_num].size!= 0)
					{
						for (int _i = 0; _i < block_list[_num].size; _i++)
						{
							double x_[3] = {data_information[frame][block_list[_num].inside[_i]][0], 
												  data_information[frame][block_list[_num].inside[_i]][1],
												  data_information[frame][block_list[_num].inside[_i]][2]};
							if (find(my_neighbor.begin(), my_neighbor.end(), block_list[_num].inside[_i]) != my_neighbor.end()) 
								continue;
							else if (distance(position, x_) <= radius)
							{
								my_neighbor.push_back(block_list[_num].inside[_i]);
							}
						}
					}
				}
			}
		}
		return my_neighbor;
}



void find_neigbhor_index(const int& frame, const int& number, double& radius)
{
	bool flag = false;
	/* compute how many blocks should be searched along each dimension */
	do
	{
		int ratio[3] = {ceil(radius/para.grid_size[0]),
						ceil(radius/para.grid_size[1]),
						ceil(radius/para.grid_size[2])};
		int index[3];

		index[0] = floor((data_information[frame][number][0] - para.RANGE[0][0])/para.grid_size[0]);
		index[1] = floor((data_information[frame][number][1] - para.RANGE[1][0])/para.grid_size[1]);
		index[2] = floor((data_information[frame][number][2] - para.RANGE[2][0])/para.grid_size[2]);
		int x_lower = max(index[0] - ratio[0],0); 
		int x_upper = min(index[0] + ratio[0] + 1, para.grid[0]);
		int y_lower = max(index[1] - ratio[1],0); 
		int y_upper = min(index[1] + ratio[1] + 1, para.grid[1]);
		int z_lower = max(index[2] - ratio[2],0);
		int z_upper = min(index[2] + ratio[2] + 1, para.grid[2]);

		for (int i = x_lower; i < x_upper; i++)
		{
			for (int j = y_lower; j < y_upper; j++)
			{
				for (int k = z_lower; k < z_upper; k++)
				{
					const int& _num = para.grid[0]*para.grid[1]*k + para.grid[0]*j + i;
					assert(_num >= 0 && _num < block_list.size());
					//const vector<int>& inside = vector<int>(block_list[_num].inside, block_list[_num].inside+block_list[_num].size);
					if (block_list[_num].size!= 0)
					{
						for (int _i = 0; _i < block_list[_num].size; _i++)
						{
							if (block_list[_num].inside[_i] == number || find(neighbor[number].begin(), 
								neighbor[number].end(), block_list[_num].inside[_i]) != neighbor[number].end()) 
								/* exclude the particle itself when counting its neighbor */
								continue;

							else if (distance(data_information[frame][number], data_information[frame][block_list[_num].inside[_i]]) <= radius)
							{
								neighbor[number].push_back(block_list[_num].inside[_i]);
							}
						}
					}
				}
			}
		}
		if (neighbor[number].size() >= THRESHOLD)
			flag = true;
		else
			radius *= TIMES;

	}while(!flag);
}



const double distance(double *a, double *b)
{
	return sqrt(pow(a[0]-b[0],2.0) + pow(a[1]-b[1],2.0) + pow(a[2]-b[2],2.0));
}



void compute_FTLE(const int& frame) //frame starts from 1 to FRAME - T
{
	/*compute previous frame and current frame */
	const int& prev_ = frame;
	const int& curr_ = prev_ + T;
	double **previous[PARTICLE];
	double **current[PARTICLE];



	double velocity_norm[PARTICLE];
	double **velocity_diff = new double*[PARTICLE];
	double ***velocity_diff_vec = new double**[PARTICLE];
	double **density_diff = new double*[PARTICLE];

	/*compute frame arrays and current arrays */
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < PARTICLE; i++)
	{
		vector<int> vec = neighbor[i];
		previous[i] = new double *[vec.size()];
		current[i] = new double *[vec.size()];

		velocity_norm[i] = sqrt(data_information[prev_][i][3]*data_information[prev_][i][3] + data_information[prev_][i][4]*data_information[prev_][i][4]
								+ data_information[prev_][i][5]*data_information[prev_][i][5]);

		density_diff[i] = new double[vec.size()];
		
		velocity_diff_vec[i] = new double*[vec.size()];

		for (int j = 0; j < vec.size(); j++)
		{
			/* compute position difference for previous and current time step */
			previous[i][j] = new double[3];
			current[i][j] = new double[3];
			velocity_diff_vec[i][j] = new double[3];
			for (int k = 0; k < 3; k++)
			{
				previous[i][j][k] = data_information[prev_][vec[j]][k] - data_information[prev_][i][k];
				current[i][j][k] = data_information[curr_][vec[j]][k] - data_information[curr_][i][k];
				velocity_diff_vec[i][j][k] = data_information[prev_][vec[j]][k] - data_information[prev_][i][k];
			}

			density_diff[i][j] = data_information[prev_][vec[j]][6] - data_information[prev_][i][6];
		}

		double result = 0.0;



		/***************     Compute rotation for each window size                ******************************/
		for(int j = 0; j < T; j++)
		{
			glm::highp_dvec3 v_1(data_information[prev_+j][i][3], data_information[prev_+j][i][4], data_information[prev_+j][i][5]);
			glm::highp_dvec3 v_2(data_information[prev_+j+1][i][3], data_information[prev_+j+1][i][4], data_information[prev_+j+1][i][5]);
			const double& dot_ = glm::dot(v_1,v_2);
			const double& cross_ = glm::length(glm::cross(v_1,v_2));
			if(abs(dot_)/glm::length(v_1)/glm::length(v_2) < 1.0e-8)
				result += acos(0.0);
			else
				result += atan(cross_/dot_);
		}
		if(isinf(result))
		{
			cout << "Found nan in rotational field computation!" << endl;
			exit(-1);
		}
		data_information[prev_ ][i][15] = (abs(result) < ERROR)? 0.0: result;
	}



	/*************** Compute FTLE value for each particles of frame, with \phi = \frac{\log(\sqrt(\phi(A*A^{T})))} ********************/
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < PARTICLE; i++)
	{
		vector<int> vec = neighbor[i];
		double array[3][3];
		//double array_[3][3];
		for (int j = 0; j < 3; j++)
		{
		/* build a linear equation ax = b; */

			double b_1 = 0.0, b_2 = 0.0, b_3 = 0.0;
			double a_11 = 0.0, a_12 = 0.0, a_13 = 0.0,
				   a_21 = 0.0, a_22 = 0.0, a_23 = 0.0,
				   a_31 = 0.0, a_32 = 0.0, a_33 = 0.0;
			Matrix3d A;


			double c_1 = 0.0, c_2 = 0.0, c_3 = 0.0;

			for (int k = 0; k < vec.size(); k++)
			{
				b_1 += current[i][k][j]*previous[i][k][0];
				b_2 += current[i][k][j]*previous[i][k][1];
				b_3 += current[i][k][j]*previous[i][k][2];

				c_1 += velocity_diff_vec[i][k][j]*previous[i][k][0];
				c_2 += velocity_diff_vec[i][k][j]*previous[i][k][1];
				c_3 += velocity_diff_vec[i][k][j]*previous[i][k][2];

				a_11 += previous[i][k][0] * previous[i][k][0];
				a_12 += previous[i][k][1] * previous[i][k][0];
				a_13 += previous[i][k][2] * previous[i][k][0];
				a_21 += previous[i][k][0] * previous[i][k][1];
				a_22 += previous[i][k][1] * previous[i][k][1];
				a_23 += previous[i][k][2] * previous[i][k][1];
				a_31 += previous[i][k][0] * previous[i][k][2];
				a_32 += previous[i][k][1] * previous[i][k][2];
				a_33 += previous[i][k][2] * previous[i][k][2];
			}
			
			/* we should use library to solve the linear equation 
			  a_11        a_12       a_13         x_1        b_1
		    [ a_21        a_22       a_23 ] *  [  x_2  ] = [ b_2  ] 
			  a_31        a_32       a_33         x_3        b_3
			  */	
			A << a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32, a_33;
			Vector3d B(b_1, b_2, b_3);
			Vector3d x = A.colPivHouseholderQr().solve(B);
			array[j][0] = x[0], array[j][1] = x[1], array[j][2] = x[2];		


			Vector3d C(c_1, c_2, c_3);
			Vector3d y = A.colPivHouseholderQr().solve(C);
			//array_[j][0] = y[0], array_[j][1] = y[1], array_[j][2] = y[2];
			data_information[prev_][i][16+j*3] = (abs(y[0])<ERROR?0:y[0]);
			data_information[prev_][i][16+j*3+1] = (abs(y[1])<ERROR?0:y[1]);
			data_information[prev_][i][16+j*3+2] = (abs(y[2])<ERROR?0:y[2]);
		}	

		Matrix3d solution_;
		solution_ << array[0][0], array[0][1], array[0][2],
					array[1][0], array[1][1], array[1][2],
					array[2][0], array[2][1], array[2][2];  
		Matrix3d solution = solution_.transpose() * solution_;	
		/* compute maximal eigenvalue for matrix */
		EigenSolver<Matrix3d> result(solution); //result is the complex vector of eigenvalues for matrix solution
		const double& lambda_1 = result.eigenvalues()[0].real();
		const double& lambda_2 = result.eigenvalues()[1].real();
		const double& lambda_3 = result.eigenvalues()[2].real();
		const double& max_eigen = lambda_1 > lambda_2?(lambda_1 > lambda_3?lambda_1:lambda_3):(lambda_2>lambda_3?lambda_2:lambda_3);
		const double& ftle = log(sqrt(max_eigen))/T;
		if(isinf(ftle))
		{
			cout << "gets inf value for particle " << i << " who has " << neighbor[i].size() << " neighboring particles!";
			cout << "The neighbor index is: " << endl;
			for (int j = 0; j < neighbor[i].size(); ++j)
			{
				cout << neighbor[i][j] << " ";
			}
			cout << endl;
			cout << "Its position is at: " << data_information[frame][i][0] << " " << data_information[frame][i][1] 
			     << " " << data_information[frame][i][2] << endl;
			cout << "Current position difference is: " << endl;
			for (int j = 0; j < neighbor[i].size(); ++j)
			{
				cout << previous[i][j][0] << " " << previous[i][j][1] << " " << previous[i][j][2] << endl;
			}

			cout << "Future position difference is: " << endl;
			for (int j = 0; j < neighbor[i].size(); ++j)
			{
				cout << current[i][j][0] << " " << current[i][j][1] << " " << current[i][j][2] << endl;
			}
			cout << "The matrix is: " << solution << endl;
			cout << "Three eigenvalues are respectively: " << endl << result.eigenvalues() << endl;
			cout << "The maximal eigenvalue is: " << max_eigen << endl;
			exit(-1);
		}
		data_information[prev_ ][i][14] = (abs(ftle) < ERROR? 0.0: ftle);

		velocity_diff[i] = new double[vec.size()];		
		for (int j = 0; j < vec.size(); j++)
		{
			velocity_diff[i][j] = velocity_norm[vec[j]] - velocity_norm[i];
			delete[] current[i][j];
			delete[] velocity_diff_vec[i][j];
		}

		delete[] current[i];
		delete[] velocity_diff_vec[i];

	}
	delete[] velocity_diff_vec;

#pragma omp parallel for num_threads(8)
	for(int i = 0; i < PARTICLE; i++)
	{
		vector<int> vec = neighbor[i];
		double array[3][3], array_[3][3];

		/* build a linear equation ax = b to compute velocity_norm gradient */

			double b_1 = 0.0, b_2 = 0.0, b_3 = 0.0;
			double a_11 = 0.0, a_12 = 0.0, a_13 = 0.0,
				   a_21 = 0.0, a_22 = 0.0, a_23 = 0.0,
				   a_31 = 0.0, a_32 = 0.0, a_33 = 0.0;
			Matrix3d A;

		/* build a linear equation ax = d to compute density graident */

			double d_1 = 0.0, d_2 = 0.0, d_3 = 0.0;

			for (int j = 0; j < vec.size(); j++)
			{
				b_1 += velocity_diff[i][j]*previous[i][j][0];
				b_2 += velocity_diff[i][j]*previous[i][j][1];
				b_3 += velocity_diff[i][j]*previous[i][j][2];

				a_11 += previous[i][j][0]*previous[i][j][0];
				a_12 += previous[i][j][1]*previous[i][j][0];
				a_13 += previous[i][j][2]*previous[i][j][0];
				a_21 += previous[i][j][0]*previous[i][j][1];
				a_22 += previous[i][j][1]*previous[i][j][1];
				a_23 += previous[i][j][2]*previous[i][j][1];
				a_31 += previous[i][j][0]*previous[i][j][2];
				a_32 += previous[i][j][1]*previous[i][j][2];
				a_33 += previous[i][j][2]*previous[i][j][2];

				d_1 += density_diff[i][j]*previous[i][j][0];
				d_2 += density_diff[i][j]*previous[i][j][1];
				d_3 += density_diff[i][j]*previous[i][j][2];
			}
			
			/* we should use library to solve the linear equation 
			  a_11        a_12       a_13         x_1        b_1
		    [ a_21        a_22       a_23 ] *  [  x_2  ] = [ b_2  ] 
			  a_31        a_32       a_33         x_3        b_3


			and

			  a_11        a_12       a_13         y_1        d_1
		    [ a_21        a_22       a_23 ] *  [  y_2  ] = [ d_2  ] 
			  a_31        a_32       a_33         y_3        d_3
			  */	
			A << a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32, a_33;
			Vector3d B(b_1, b_2, b_3);
			Vector3d x = A.colPivHouseholderQr().solve(B);
			data_information[prev_][i][8] = (abs(x[0]) < ERROR? 0: x[0]);
			data_information[prev_][i][9] = (abs(x[1]) < ERROR? 0: x[1]);
			data_information[prev_][i][10] = (abs(x[2]) < ERROR? 0: x[2]);	


			Vector3d C(d_1, d_2, d_3);
			Vector3d y = A.colPivHouseholderQr().solve(C);	
			data_information[prev_][i][11] = (abs(y[0]) < ERROR?0:y[0]);
			data_information[prev_][i][12] = (abs(y[1]) < ERROR?0:y[1]);
			data_information[prev_][i][13] = (abs(y[2]) < ERROR?0:y[2]);	
			for (int k = 0; k < vec.size(); k++)
			{
				delete[] previous[i][k];
			}

			delete[] previous[i];
			delete[] velocity_diff[i];
			delete[] density_diff[i];
	}

	delete[] velocity_diff;
	delete[] density_diff;
}
