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
const int FRAME = 148;
const int PARTICLE = 128000;

const int NUMBER = 148;
double **data_information[NUMBER]; //this will store information from files
vector< vector<int > > neighbor(PARTICLE); //this will store neighbor of the current frame
vector< Hashing::Block> block_list;
const int& T = 20;
const double& radius = 0.1;
Hashing::Parameter para;


const double& TIMES = 1.1;
const int& THRESHOLD = 15;

int times = 0;

const double distance(double *a, double *b);
void initialize_memory_file_reader();
void release_memory_file_reader();
void load_data(const int& start);
void find_neigbhor_index(const int& frame, const int& number, double& radius);
void find_neighbor(const int& frame, const double& radius);
void my_init();
void assigned_grid(const int& frame);
void compute_FTLE(const int& frame);
double extrema[2] = {INT_MAX, INT_MIN};
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
int main(int argc, char **argv)
{
	double io_time = 0;
	clock_t starting, ending;
	time(&starting);
	initialize_memory_file_reader();
	
	/* will perform 7 times of I-O operations */
	for (int i = 0; i < ceil((double)FRAME/(double)NUMBER); i++)
	{
		clock_t begin, end;
		time(&begin);
		/* loading 305 files every time from 1 + 300*i */
		//load_data(1 + (NUMBER - T)*i);
		
		load_data((NUMBER - T)*i);
		time(&end);
		io_time += (double)difftime(end, begin);

		for (int j = 1; j < NUMBER - T + 1; j++)//compute at most NUMBER-T+1 FTLE values
		{
			const int& frame = j + i*(NUMBER - T); //frame will be from 1 -> 1000 - 5
			if (frame > FRAME - T)
			{
				break;
			}
			cout << "Frame " << frame << " computation for FTLE starts!" << endl;

			assigned_grid(j);			
			find_neighbor(j, radius);
			block_list.clear();
			compute_FTLE(j);
			string frame_str;
			stringstream ss;
			ss << frame;
			ss >> frame_str;
			frame_str = string("FTLE frame ") + frame_str + string(".vtk");
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
			fout << "SCALARS ftle float 1" << endl;
			fout << "LOOKUP_TABLE ftletable" << endl;
			for ( int k = 0; k < PARTICLE; k++)
			{
				fout << data_information[j][k][3] << endl;
				if (data_information[j][k][3] < extrema[0])
				{
					extrema[0] = data_information[j][k][3];
				}
				if (data_information[j][k][3] > extrema[1])
				{
					extrema[1] = data_information[j][k][3];
				}
			}	
			fout.close();
			cout << "Frame " << frame << " computation ends!" << endl;
		}
	}

	release_memory_file_reader();
	time(&ending);

	ofstream out("README", ios::out);
	out << "Total computational time is: " << double(difftime(ending, starting)) << " S!" << endl;
	out << "I-O time is: " << io_time << " S" << endl;
	out << "FTLE value range is: " << extrema[0] << ", " << extrema[1] << endl;
	out.close();
	return 0;
}




void initialize_memory_file_reader()
/* memory allocation for data_information to store double[305][128000][4] */
{
#pragma omp parallel for num_threads(4)

	for(int i = 0; i < NUMBER; i++)
	{
		data_information[i] = new double *[PARTICLE];
		for (int j = 0; j < PARTICLE; j++)
		{
			data_information[i][j] = new double[5];
		}
	}
	std::cout << "Memory allocation completed!" << std::endl;
	std::cout << std::endl;
}



void release_memory_file_reader()
{
#pragma omp parallel for num_threads(4)

	for (int i = 0; i < NUMBER; i++)
	{
		for (int j = 0; j < PARTICLE; j++)
		{
			delete[] data_information[i][j];
		}
		delete[] data_information[i];
	}

#pragma omp parallel for num_threads(4)
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
	const string& location = "/home/lieyu/Downloads/experiement_data/PBF_paralized_code_learnt_from_PBD/vorticity0.01_epsilon6_time0.003_iteration5_interval0.05/source_data/";


#pragma omp parallel for num_threads(4)
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
		double a,b,c,d;
		for (int j = 0; j < PARTICLE; j++)
		{
			in_ >> data_information[i-start][j][0] >> data_information[i-start][j][1] >> data_information[i-start][j][2];
			in_ >> a >> b >> c >> d;
		}
		ss.clear();
		in_.close();
		cout << "File " << (i) << " has been loaded successfully!" << endl;
	}

	cout << "Finished I-O Operations!" << endl;
	times++;
}




void assigned_grid(const int& frame)
{

	double range[3][2] = {INT_MAX, INT_MIN, 
						  INT_MAX, INT_MIN, 
						  INT_MAX, INT_MIN};

	for (int i = 0; i < PARTICLE; i++)
	{
		if (range[0][0] > data_information[frame-1][i][0])
		{
			range[0][0] = data_information[frame-1][i][0];
		}
		else if (range[0][1] < data_information[frame-1][i][0])
		{
			range[0][1] = data_information[frame-1][i][0];
		}
		if (range[1][0] > data_information[frame-1][i][1])
		{
			range[1][0] = data_information[frame-1][i][1];
		}
		else if (range[1][1] < data_information[frame-1][i][1])
		{
			range[1][1] = data_information[frame-1][i][1];
		}
		if (range[2][0] > data_information[frame-1][i][2])
		{
			range[2][0] = data_information[frame-1][i][2];
		}
		else if (range[2][1] < data_information[frame-1][i][2])
		{
			range[2][1] = data_information[frame-1][i][2];
		}
	}

	/*   compute the member variables for para (Hashing::Parameter)   */
	para.compute_grid(range);
	cout << "Range is: [ " << para.RANGE[0][0] << ", " << para.RANGE[0][1] << "] X [ " <<
		para.RANGE[1][0] << ", " << para.RANGE[1][1] << "] X [ " << para.RANGE[2][0] << ", "
		<< para.RANGE[2][1] << "]." << endl;

	const int& b_size = para.grid[0]*para.grid[1]*para.grid[2];
	block_list = std::vector<Hashing::Block >(b_size);
	const int& size = block_list.size();
	
#pragma omp parallel for num_threads(4)
	/* assign each particle to its relative block */
	for (int i = 0; i < PARTICLE; i++)
	{
		const int& a = floor((data_information[frame-1][i][0] - para.RANGE[0][0])/para.grid_size[0]);
		const int& b = floor((data_information[frame-1][i][1] - para.RANGE[1][0])/para.grid_size[1]);
		const int& c = floor((data_information[frame-1][i][2] - para.RANGE[2][0])/para.grid_size[2]);
		const int& index = a + b*para.grid[0] + c*para.grid[0]*para.grid[1];
		assert(index >= 0 && index < size);
		/*cout << "Particle " << i << " has position at " << data_information[frame-1][i][0] << " "
		     << data_information[frame-1][i][1] << " " << data_information[frame-1][i][2] << endl;*/
		block_list[index].add(i);
	}
	cout << "Frame " << frame << " assigning particle positions completed!" << endl;
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
		data_information[frame-1][k][3] = radius;
		find_neigbhor_index(frame, k, data_information[frame-1][k][3]);
	}

	/*stringstream ss;
	string frame_str;
	ss << frame;
	ss >> frame_str;
	ofstream out_(("PBF_position " + frame_str + ".txt").c_str(), ios::out);
	if (!out_)
	{
		cout << "Error creating a position file!" << endl;
		exit(-1);
	}
	for (int i = 0; i < PARTICLE; ++i)
	{
		out_ << data_information[frame-1][i][0] << " " << data_information[frame-1][i][1]
		<< " " << data_information[frame-1][i][2] << " " << data_information[frame-1][i][3] 
		<< " " << neighbor[i].size() << endl;
	}
	out_.close();*/
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

		index[0] = floor((data_information[frame-1][number][0] - para.RANGE[0][0])/para.grid_size[0]);
		index[1] = floor((data_information[frame-1][number][1] - para.RANGE[1][0])/para.grid_size[1]);
		index[2] = floor((data_information[frame-1][number][2] - para.RANGE[2][0])/para.grid_size[2]);
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

							else if (distance(data_information[frame-1][number], data_information[frame-1][block_list[_num].inside[_i]]) <= radius)
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

	/*compute frame arrays and current arrays */
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < PARTICLE; i++)
	{
		vector<int> vec = neighbor[i];
		previous[i] = new double *[vec.size()];
		current[i] = new double *[vec.size()];
		
		for (int j = 0; j < vec.size(); j++)
		{
			/* compute position difference for previous and current time step */
			previous[i][j] = new double[3];
			current[i][j] = new double[3];
			for (int k = 0; k < 3; k++)
			{
				previous[i][j][k] = data_information[prev_-1][vec[j]][k] - data_information[prev_-1][i][k];
				current[i][j][k] = data_information[curr_-1][vec[j]][k] - data_information[curr_-1][i][k];
			}
		}
	}

	/*************** Compute FTLE value for each particles of frame, with \phi = \frac{\log(\sqrt(\phi(A*A^{T})))} ********************/
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < PARTICLE; i++)
	{
		vector<int> vec = neighbor[i];
		double array[3][3];
		for (int j = 0; j < 3; j++)
		{
		/* build a linear equation ax = b; */

			double b_1 = 0.0, b_2 = 0.0, b_3 = 0.0;
			double a_11 = 0.0, a_12 = 0.0, a_13 = 0.0,
				   a_21 = 0.0, a_22 = 0.0, a_23 = 0.0,
				   a_31 = 0.0, a_32 = 0.0, a_33 = 0.0;
			Matrix3d A;
			for (int k = 0; k < vec.size(); k++)
			{
				b_1 += current[i][k][j]*previous[i][k][0];
				b_2 += current[i][k][j]*previous[i][k][1];
				b_3 += current[i][k][j]*previous[i][k][2];
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
			cout << "Its position is at: " << data_information[frame-1][i][0] << " " << data_information[frame-1][i][1] 
			     << " " << data_information[frame-1][i][2] << endl;
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

		for (int k = 0; k < vec.size(); k++)
		{
			delete[] previous[i][k];
			delete[] current[i][k];
		}

		delete[] previous[i];
		delete[] current[i];
		data_information[prev_ - 1][i][3] = (abs(ftle) < 1.0e-5)? 0.0: ftle;
	}
}
