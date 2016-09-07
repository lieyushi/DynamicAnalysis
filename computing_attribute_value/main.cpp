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
using namespace std;
using namespace Eigen;
const int FRAME = 149;
const int COLUMN = 15;
const int PARTICLE = 128000;
double data_information[PARTICLE][COLUMN];

vector< vector<int > > neighbor(PARTICLE); //this will store neighbor of the current frame
vector< Hashing::Block> block_list;

double radius = 0.1;
Hashing::Parameter para;

const double& TIMES = 1.1;
const int& THRESHOLD = 15;

const double distance(double *a, double *b);
void release_memory_file_reader();
void load_data(const int& start);
void find_neigbhor_index(const int& number, double& radius);
void find_neighbor(const int& frame, const double& radius);
void assigned_grid(const int& frame);
void compute_gradient(const int& frame);
void output(const int& frame);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
int main(int argc, char **argv)
{
	clock_t starting, ending;
	time(&starting);
	for (int i = 0; i < FRAME; i++)
	{
		load_data(i);
		assigned_grid(i);
		find_neighbor(i, radius);
		compute_gradient(i);
		output(i);
	}	
	release_memory_file_reader();
	time(&ending);
	ofstream out("README", ios::out);
	out << "Total computational time is: " << double(difftime(ending, starting)) << " S!" << endl;
	out.close();
	return 0;
}

void output(const int& frame)
{
	stringstream ss;
	string frame_;
	ss << frame;
	ss >> frame_;
	ofstream fout(("Attribute " + frame_ + ".txt").c_str(), ios::out);
	if(fout.fail())
	{
		cout << "Error creating files!" << endl;
		exit(-1);
	}

	for (int i = 0; i < PARTICLE; ++i)
	{
		for (int j = 0; j < COLUMN; ++j)
		{
			fout << data_information[i][j] << " ";
		}
		fout << endl;
	}
	fout.close();
}


void release_memory_file_reader()
{
	block_list.clear();
	std::cout << "Memory elimination completed! " << std::endl;
	std::cout << std::endl;
}


void load_data(const int& start) 
{
	const string& location = "/home/lieyu/Downloads/experiement_data/PBF_paralized_code_learnt_from_PBD/vorticity0.01_epsilon6_time0.003_iteration5_interval0.05/source_data/";
		string file_;
		stringstream ss;
		ss << start;
		ss >> file_;		
		file_ = location + string("Frame ") + file_ + string(".txt");
		ifstream in_(file_.c_str(), ios::in);
		if (!in_)
		{
			cout << "File doesn't exist in designated directory!" << endl;
			exit(1);
		}

		for (int j = 0; j < PARTICLE; j++)
		{
			in_ >> data_information[j][0] >> data_information[j][1] >> data_information[j][2] >> data_information[j][3] >> data_information[j][4]
			    >> data_information[j][5] >> data_information[j][6];
		}
		ss.clear();
		in_.close();
		cout << "File " << start << " has been loaded successfully!" << endl;
}




void assigned_grid(const int& frame)
{

	double range[3][2] = {INT_MAX, INT_MIN, 
						  INT_MAX, INT_MIN, 
						  INT_MAX, INT_MIN};

	for (int i = 0; i < PARTICLE; i++)
	{
		if (range[0][0] > data_information[i][0])
		{
			range[0][0] = data_information[i][0];
		}
		else if (range[0][1] < data_information[i][0])
		{
			range[0][1] = data_information[i][0];
		}
		if (range[1][0] > data_information[i][1])
		{
			range[1][0] = data_information[i][1];
		}
		else if (range[1][1] < data_information[i][1])
		{
			range[1][1] = data_information[i][1];
		}
		if (range[2][0] > data_information[i][2])
		{
			range[2][0] = data_information[i][2];
		}
		else if (range[2][1] < data_information[i][2])
		{
			range[2][1] = data_information[i][2];
		}
	}

	/*   compute the member variables for para (Hashing::Parameter)   */
	para.compute_grid(range);
	cout << "Range is: [ " << para.RANGE[0][0] << ", " << para.RANGE[0][1] << "] X [ " <<
		para.RANGE[1][0] << ", " << para.RANGE[1][1] << "] X [ " << para.RANGE[2][0] << ", "
		<< para.RANGE[2][1] << "]." << endl;

	const int& b_size = para.grid[0]*para.grid[1]*para.grid[2];
	block_list.clear();
	block_list = std::vector<Hashing::Block >(b_size);
	const int& size = block_list.size();
	
#pragma omp parallel for num_threads(4)
	/* assign each particle to its relative block */
	for (int i = 0; i < PARTICLE; i++)
	{
		const int& a = floor((data_information[i][0] - para.RANGE[0][0])/para.grid_size[0]);
		const int& b = floor((data_information[i][1] - para.RANGE[1][0])/para.grid_size[1]);
		const int& c = floor((data_information[i][2] - para.RANGE[2][0])/para.grid_size[2]);
		const int& index = a + b*para.grid[0] + c*para.grid[0]*para.grid[1];
		assert(index >= 0 && index < size);
		/*cout << "Particle " << i << " has position at " << data_information[frame-1][i][0] << " "
		     << data_information[frame-1][i][1] << " " << data_information[frame-1][i][2] << endl;*/
		block_list[index].add(i);
	}

	cout << "Frame " << frame << " has been allocated in grid!" << endl;
}




void find_neighbor(const int& frame, const double& radius)
{
#pragma omp parallel for num_threads(4)
	/* firstly clear the former data in its neighbor */
	for (int i = 0; i < PARTICLE; i++)
	{
		neighbor[i].clear();
	}

#pragma omp parallel for num_threads(4)	
	/* invoke neighbor information for each particle */
	for (int k = 0; k < PARTICLE; k++)
	{
		data_information[k][7] = radius;
		find_neigbhor_index(k, data_information[k][7]);
		data_information[k][8] = neighbor[k].size();
	}

	cout << "Frame " << frame << " neighbor-searching finished!" << endl;
}


void find_neigbhor_index(const int& number, double& radius)
{
	bool flag = false;
	/* compute how many blocks should be searched along each dimension */
	do
	{
		int ratio[3] = {ceil(radius/para.grid_size[0]),
						ceil(radius/para.grid_size[1]),
						ceil(radius/para.grid_size[2])};
		int index[3];

		index[0] = floor((data_information[number][0] - para.RANGE[0][0])/para.grid_size[0]);
		index[1] = floor((data_information[number][1] - para.RANGE[1][0])/para.grid_size[1]);
		index[2] = floor((data_information[number][2] - para.RANGE[2][0])/para.grid_size[2]);
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

					if (block_list[_num].size!= 0)
					{
						for (int _i = 0; _i < block_list[_num].size; _i++)
						{
							if (block_list[_num].inside[_i] == number || find(neighbor[number].begin(), 
								neighbor[number].end(), block_list[_num].inside[_i]) != neighbor[number].end()) 
								/* exclude the particle itself when counting its neighbor */
								continue;

							else if (distance(data_information[number], data_information[block_list[_num].inside[_i]]) <= radius)
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



void compute_gradient(const int& frame)
{
	double velocity_norm[PARTICLE];
	double **velocity_diff = new double*[PARTICLE];
	double **density_diff = new double*[PARTICLE];
	double ***position_difference = new double**[PARTICLE]; 

	/*compute frame arrays and current arrays */
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < PARTICLE; i++)
	{
		vector<int> vec = neighbor[i];
		position_difference[i] = new double *[vec.size()];
		velocity_norm[i] = sqrt(data_information[i][3]*data_information[i][3] + data_information[i][4]*data_information[i][4]
								+ data_information[i][5]*data_information[i][5]);

		density_diff[i] = new double[vec.size()];
		
		for (int j = 0; j < vec.size(); j++)
		{
			/* compute position difference for previous and current time step */
			position_difference[i][j] = new double[3];
			for (int k = 0; k < 3; k++)
			{
				position_difference[i][j][k] = data_information[vec[j]][k] - data_information[i][k];
			}

			density_diff[i][j] = data_information[vec[j]][6] - data_information[i][6];
		}
	}

#pragma omp parallel for num_threads(4)
	for (int i = 0; i < PARTICLE; i++)
	{
		vector<int> vec = neighbor[i];
		velocity_diff[i] = new double[vec.size()];		
		for (int j = 0; j < vec.size(); j++)
		{
			velocity_diff[i][j] = velocity_norm[vec[j]] - velocity_norm[i];
		}
	}




	/*************** Compute FTLE value for each particles of frame, with \phi = \frac{\log(\sqrt(\phi(A*A^{T})))} ********************/
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < PARTICLE; i++)
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
				b_1 += velocity_diff[i][j]*position_difference[i][j][0];
				b_2 += velocity_diff[i][j]*position_difference[i][j][1];
				b_3 += velocity_diff[i][j]*position_difference[i][j][2];

				a_11 += position_difference[i][j][0]*position_difference[i][j][0];
				a_12 += position_difference[i][j][1]*position_difference[i][j][0];
				a_13 += position_difference[i][j][2]*position_difference[i][j][0];
				a_21 += position_difference[i][j][0]*position_difference[i][j][1];
				a_22 += position_difference[i][j][1]*position_difference[i][j][1];
				a_23 += position_difference[i][j][2]*position_difference[i][j][1];
				a_31 += position_difference[i][j][0]*position_difference[i][j][2];
				a_32 += position_difference[i][j][1]*position_difference[i][j][2];
				a_33 += position_difference[i][j][2]*position_difference[i][j][2];

				d_1 += density_diff[i][j]*position_difference[i][j][0];
				d_2 += density_diff[i][j]*position_difference[i][j][1];
				d_3 += density_diff[i][j]*position_difference[i][j][2];
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
			data_information[i][9] = x[0], data_information[i][10] = x[1], data_information[i][11] = x[2];	


			Vector3d C(d_1, d_2, d_3);
			Vector3d y = A.colPivHouseholderQr().solve(C);	
			data_information[i][12] = y[0], data_information[i][13] = y[1], data_information[i][14] = y[2];	


		
		for (int k = 0; k < vec.size(); k++)
		{
			delete[] position_difference[i][k];
		}

		delete[] position_difference[i];
		delete[] velocity_diff[i];
		delete[] density_diff[i];
	}

	delete[] position_difference;
	delete[] velocity_diff;
	delete[] density_diff;
}
