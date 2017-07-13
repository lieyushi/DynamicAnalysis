#include "FTLE_Calculator.h"
#include <algorithm>
#include <omp.h>

#define ERROR 1.0e-8

const float& TIMES = 1.1;
const float& INITIAL = 1.1;
const int& THRESHOLD = 8;
const float& stepSize = 2.0/float(UNIT)*PI;
const static int& N = UNIT+1;

FTLE_Calculator::FTLE_Calculator(void)
{
	para = Hashing::Parameter();
}


FTLE_Calculator::~FTLE_Calculator(void)
{
	neighbor.clear();
}


void FTLE_Calculator::get_FTLE_value(float **data_information, const int& FRAME, const int& PARTICLE, const int& T,
									 const float& time_step, const int& choice, float readme[][2])
{
	neighbor = std::vector< std::vector<int > >(PARTICLE);

	for (int j = 0; j < FRAME - T; j++)
	{
		cout << "Frame " << j << " computation for FTLE starts!" << endl;

		ABC_flow::generate_flow_velocity(data_information, j, PARTICLE, time_step, choice, T);
		assigned_grid(data_information, j, PARTICLE);	 		
		find_neighbor(data_information, PARTICLE);
		compute_FTLE(data_information, PARTICLE, T, time_step, choice, j);
		get_limit(data_information, PARTICLE, readme);
		release_neighbor();
		block_list.clear();

		string frame_str;
		stringstream ss;
		ss << j;
		ss >> frame_str;
		frame_str = string("FTLE frame ") + frame_str + string(".txt");
		ss.clear();
		ofstream out(frame_str.c_str(), ios::out);
		if (!out)
		{
			cout << "Error creating the file!" << endl;
			exit(-1);
		}
		float *temp;
		for (int k = 0; k < PARTICLE; k++)
		{
			temp = data_information[k];
			for (int k_ = 0; k_ < 72; ++k_)
			{
				out << temp[k_] << " ";
			}
			out << endl;
		}
		out.close();

		generate_full_vtk(data_information, j, PARTICLE);
		generate_interior_vtk(data_information, j, PARTICLE);
		cout << "Frame " << j << " computation ends!" << endl;
		
	}

}



void FTLE_Calculator::release_neighbor()
{
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < neighbor.size(); ++i)
	{
		neighbor[i].clear();
	}
}


/*------------------------ assign all particles into the blocks based on their positions --------------------------*/
void FTLE_Calculator::assigned_grid(float **data_information, const int& frame, const int& PARTICLE)
{
	
	/****** This is to compute the domain to be decomposed *********/

	// compute minimal and maximal values for x, y and z
	float range[3][2] = {
							FLT_MAX, FLT_MIN, 
							FLT_MAX, FLT_MIN, 
							FLT_MAX, FLT_MIN
						};

	float *temp = NULL;
	for (int i = 0; i < PARTICLE; i++)
	{
		temp = data_information[i];
		if (range[0][0] > temp[0])
		{
			range[0][0] = temp[0];
		}
		else if (range[0][1] < temp[0])
		{
			range[0][1] = temp[0];
		}
		if (range[1][0] > temp[1])
		{
			range[1][0] = temp[1];
		}
		else if (range[1][1] < temp[1])
		{
			range[1][1] = temp[1];
		}
		if (range[2][0] > temp[2])
		{
			range[2][0] = temp[2];
		}
		else if (range[2][1] < temp[2])
		{
			range[2][1] = temp[2];
		}
	}

	/*   compute the member variables for para (Hashing::Parameter)   */
	para.compute_grid(range);

	cout << "Range is: [ " << para.RANGE[0][0] << ", " << para.RANGE[0][1] << "] X [ " <<
		para.RANGE[1][0] << ", " << para.RANGE[1][1] << "] X [ " << para.RANGE[2][0] << ", "
		<< para.RANGE[2][1] << "]." << endl;

	// preset 50*50*50 = 125000 blocks for the domain decomposition
	const int& b_size = para.grid[0] * para.grid[1] * para.grid[2];
	block_list = std::vector<Hashing::Block >(b_size);

	/* determine how many block grids needed for domain decomposition */
	const int& size = block_list.size();

//#pragma omp parallel for num_threads(8)
	/* assign each particle to its relative block */
	for (int i = 0; i < PARTICLE; i++)
	{
		temp = data_information[i];
		const int& a = floor((temp[0] - para.RANGE[0][0])/para.grid_size[0]);
		const int& b = floor((temp[1] - para.RANGE[1][0])/para.grid_size[1]);
		const int& c = floor((temp[2] - para.RANGE[2][0])/para.grid_size[2]);
		assert(a >= 0 && a < para.grid[0]);
		assert(b >= 0 && b < para.grid[1]);
		assert(c >= 0 && c < para.grid[2]);
		const int& index = a + b*para.grid[0] + c*para.grid[0]*para.grid[1];
		block_list[index].inside.push_back(i);
	}
	cout << "Frame " << frame << " assigning particle positions completed!" << endl;
}


/*------------------ Find the neighbor particles for all particles -----------------------*/
void FTLE_Calculator::find_neighbor(float **data_information, const int& PARTICLE)
{

	// dynamically preset neighbor searching radius
	//const float& radius = INITIAL*get_max(para.grid_size[0], para.grid_size[1], para.grid_size[2]);
	const float& radius = 1.0/50.0*PI*INITIAL;
	// search neighbor for all particles
#pragma omp parallel for num_threads(8)	
	for (int k = 0; k < PARTICLE; k++)
	{
		float *temp = data_information[k];
		temp[6] = radius;
		find_neigbhor_index(data_information, k, temp[6]);
	}
}


void FTLE_Calculator::find_neigbhor_index(float **data_information, const int& number, float& radius)
{
	bool flag = false;
	float *temp = NULL;
	do{
		int ratio[3] = {int(ceil(radius/para.grid_size[0])),
						int(ceil(radius/para.grid_size[1])),
						int(ceil(radius/para.grid_size[2]))};
		int index[3];
		temp = data_information[number];
		index[0] = int(floor((temp[0] - para.RANGE[0][0])/para.grid_size[0]));
		index[1] = int(floor((temp[1] - para.RANGE[1][0])/para.grid_size[1]));
		index[2] = int(floor((temp[2] - para.RANGE[2][0])/para.grid_size[2]));
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
					vector<int> inside = block_list[_num].inside;
					if (inside.size()!= 0)
					{
						for (int _i = 0; _i < inside.size(); _i++)
						{
							if (inside[_i] == number || find(neighbor[number].begin(), 
								neighbor[number].end(), inside[_i]) != neighbor[number].end()) 
								continue;
							else if (distance(data_information[number], data_information[inside[_i]]) <= radius)
							{
								neighbor[number].push_back(inside[_i]);
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


void FTLE_Calculator::find_neigbhor_index(float **data_information, float *virtualPos, float& radius, std::vector<int>& virtualNeighbor)
{
	bool flag = false;
	do{
		int ratio[3] = {int(ceil(radius/para.grid_size[0])),
						int(ceil(radius/para.grid_size[1])),
						int(ceil(radius/para.grid_size[2]))};
		int index[3];
		index[0] = int(floor((virtualPos[0] - para.RANGE[0][0])/para.grid_size[0]));
		index[1] = int(floor((virtualPos[1] - para.RANGE[1][0])/para.grid_size[1]));
		index[2] = int(floor((virtualPos[2] - para.RANGE[2][0])/para.grid_size[2]));
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
					vector<int> inside = block_list[_num].inside;
					if (inside.size()!= 0)
					{
						for (int _i = 0; _i < inside.size(); _i++)
						{
							if (find(virtualNeighbor.begin(), virtualNeighbor.end(), inside[_i]) != virtualNeighbor.end()) 
								continue;
							else if (distance(virtualPos, data_information[inside[_i]]) <= radius)
							{
								virtualNeighbor.push_back(inside[_i]);
							}
						}
					}
				}
			}
		}
		if (virtualNeighbor.size() >= THRESHOLD)
			flag = true;
		else
			radius *= TIMES;
	}while(!flag);
}




const float FTLE_Calculator::distance(float *a, float *b)
{
	return sqrt(pow(a[0]-b[0],2.0) + pow(a[1]-b[1],2.0) + pow(a[2]-b[2],2.0));
}


const float getNorm(float *a)
{
	return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}


void FTLE_Calculator::compute_FTLE(float **data_information, const int& PARTICLE, const int& T,
								   const float& time_step, const int& choice, const int& frame) //frame starts from 1 to FRAME - T
{
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < PARTICLE; i++)
	{
		getEachRealJacobian(&data_information[i][0], frame, time_step, choice);
		getRealFTLE(data_information, frame, i, T, time_step, choice);
		computeCentralDifference(data_information, i, T);
		computeEachLSF_ftle(data_information, i, T);
	}
}


const float& get_max(const float& a, const float& b, const float& c)
{
	return (a>b)?((a>c)?a:c):((b>c)?b:c);
}

void FTLE_Calculator::computeCentralDifference(float **data_information, const int& index, const int& T)
{
	const int& a = index%N;
	const int& b = index/N%N;
	const int& c = index/N/N;
	float *temp, *top, *bottom, *back, *front, *left, *right;
	if(a==0||a==UNIT||b==0||b==UNIT||c==0||c==UNIT)
	{
		temp = data_information[index];
		if(a==0)
		{
			right = data_information[c*N*N+b*N+a+1];
			for (int i = 0; i < 3; ++i)
			{
				temp[49+3*i]=(right[i+3]-temp[i+3])/stepSize;
			}
		}
		else if(a==UNIT)
		{
			left = data_information[c*N*N+b*N+a-1];
			for (int i = 0; i < 3; ++i)
			{
				temp[49+3*i]=(temp[i+3]-left[i+3])/stepSize;
			}
		}
		else
		{
			left = data_information[c*N*N+b*N+a-1];
			right = data_information[c*N*N+b*N+a+1];
			for (int i = 0; i < 3; ++i)
			{
				temp[49+3*i]=(right[i+3]-left[i+3])/2.0/stepSize;
			}
		}

		if(b==0)
		{
			front = data_information[c*N*N+(b+1)*N+a];
			for (int i = 0; i < 3; ++i)
			{
				temp[50+3*i]=(front[i+3]-temp[i+3])/stepSize;
			}
		}
		else if(b==UNIT)
		{
			back = data_information[c*N*N+(b-1)*N+a];
			for (int i = 0; i < 3; ++i)
			{
				temp[50+3*i]=(temp[i+3]-back[i+3])/stepSize;
			}
		}
		else
		{
			back = data_information[c*N*N+(b-1)*N+a];
			front = data_information[c*N*N+(b+1)*N+a];
			for (int i = 0; i < 3; ++i)
			{
				temp[50+3*i]=(front[i+3]-back[i+3])/2.0/stepSize;
			}
		}

		if(c==0)
		{
			top = data_information[(c+1)*N*N+b*N+a];
			for (int i = 0; i < 3; ++i)
			{
				temp[51+3*i]=(top[i+3]-temp[i+3])/stepSize;
			}
		}
		else if(c==UNIT)
		{
			bottom = data_information[(c-1)*N*N+b*N+a];
			for (int i = 0; i < 3; ++i)
			{
				temp[51+3*i]=(temp[i+3]-bottom[i+3])/stepSize;
			}
		}
		else
		{
			top = data_information[(c+1)*N*N+b*N+a];
			bottom = data_information[(c-1)*N*N+b*N+a];
			for (int i = 0; i < 3; ++i)
			{
				temp[51+3*i]=(top[i+3]-bottom[i+3])/2.0/stepSize;
			}
		}
		return;
	}
	temp = data_information[index];
	top = data_information[(c+1)*N*N+b*N+a];
	bottom = data_information[(c-1)*N*N+b*N+a];
	back = data_information[c*N*N+(b-1)*N+a];
	front = data_information[c*N*N+(b+1)*N+a];
	left = data_information[c*N*N+b*N+a-1];
	right = data_information[c*N*N+b*N+a+1];

	for (int i = 0; i < 3; ++i)
	{
		temp[49+i*3] = (right[3+i]-left[3+i])/2.0/stepSize;
		temp[50+i*3] = (front[3+i]-back[3+i])/2.0/stepSize;
		temp[51+i*3] = (top[3+i]-bottom[3+i])/2.0/stepSize;
	}

	Matrix3f centralFTLE;
	for (int i = 0; i < 3; ++i)
	{
		centralFTLE(i,0) = (right[46+i]-left[46+i])/2.0/stepSize;
		centralFTLE(i,1) = (front[46+i]-back[46+i])/2.0/stepSize;
		centralFTLE(i,2) = (top[46+i]-bottom[46+i])/2.0/stepSize;
	}
	Matrix3f central = centralFTLE.transpose()*centralFTLE;
	EigenSolver<Matrix3f> result(central); //result is the complex vector of eigenvalues for matrix solution
	const float& lambda_1 = result.eigenvalues()[0].real();
	const float& lambda_2 = result.eigenvalues()[1].real();
	const float& lambda_3 = result.eigenvalues()[2].real();
	float max_eigen = lambda_1 > lambda_2?(lambda_1 > lambda_3?lambda_1:lambda_3):(lambda_2>lambda_3?lambda_2:lambda_3);
	const float& ftle = log(sqrt(max_eigen))/T;

#ifdef __APPLE__		
	if(_isnan(ftle))
#elif __linux__
	if(isinf(ftle))
#else
    error "Unkown compiler"
    exit(-1)
#endif
	{
		cout << "Error found in inf value for ftle computation!" << endl;
		exit(-1);
	}
	temp[58] = (abs(ftle) < ERROR)? 0.0: ftle;
}


void FTLE_Calculator::generate_full_vtk(float **data_information, const int& frame, const int& PARTICLE)
{	
	std::ofstream fout;
	string frame_str;
	stringstream ss;
	ss << frame;
	ss >> frame_str;
	ss.clear();

	fout.open( ("frame" + frame_str + " full.vtk").c_str(), ios::out);

	fout << "# vtk DataFile Version 3.0" << endl;
	fout << "Volume example" << endl;
	fout << "ASCII" << endl;
	fout << "DATASET STRUCTURED_POINTS" << endl;
	fout << "DIMENSIONS " << (N) << " " << (N) << " " << (N) << endl;
	fout << "ASPECT_RATIO " << stepSize << " " << stepSize << " " << stepSize << endl;
	fout << "ORIGIN " << 0.0 << " " << 0.0 << " " << 0.0 << endl; 

	fout << "POINT_DATA " << PARTICLE << endl;

	fout << "SCALARS Radius float 1" << endl;
	fout << "LOOKUP_TABLE radius_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][6] << endl;
	}

	fout << "SCALARS fitted_ftle float 1" << endl;
	fout << "LOOKUP_TABLE mlsf_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][7] << endl;
	}

	fout << "SCALARS RealFtle float 1" << endl;
	fout << "LOOKUP_TABLE real_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][8] << endl;
	}

	fout << "SCALARS ftleDiff_central_fitted float 1" << endl;
	fout << "LOOKUP_TABLE diff_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][9] << endl;
	}

	fout << "SCALARS JacobianDiff_Determinant float 1" << endl;
	fout << "LOOKUP_TABLE JacobiandiffDeterminant_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][28] << endl;
	}

	fout << "SCALARS JacobianDifferenceFrobinus float 1" << endl;
	fout << "LOOKUP_TABLE JacobianDiffFrobius_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][29] << endl;
	}

	fout << "SCALARS JacobianDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE JacobianDiffMax_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][30] << endl;
	}

	fout << "SCALARS realJacobianVecError float 1" << endl;
	fout << "LOOKUP_TABLE realJacobianVecError_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][31] << endl;
	}

	fout << "VECTORS velocityDirection float" << endl;
	float *temp = NULL;
	for ( int j = 0; j < PARTICLE; j++)
	{
		temp = data_information[j];
		fout << temp[3] << " " << temp[4] << " " << temp[5] << endl;
	}

	fout << "SCALARS speed float 1" << endl;
	fout << "LOOKUP_TABLE speed_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << getNorm(&data_information[j][3]) << endl;
	}

	fout << "SCALARS fewerDiff_Determinant float 1" << endl;
	fout << "LOOKUP_TABLE fewerDiffDeterminant_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][41] << endl;
	}

	fout << "SCALARS fewerDifferenceFrobinus float 1" << endl;
	fout << "LOOKUP_TABLE fewerDiffFrobius_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][42] << endl;
	}

	fout << "SCALARS fewerDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE fewerDiffMax_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][43] << endl;
	}

	fout << "SCALARS fittedJacobianVecError float 1" << endl;
	fout << "LOOKUP_TABLE fittedJacobianVecError_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][44] << endl;
	}

	fout << "SCALARS fewerJacobianVecError float 1" << endl;
	fout << "LOOKUP_TABLE fewerJacobianVecError_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][45] << endl;
	}

	fout << "SCALARS fullToCentralDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE fullToCentralDiffMax_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][59] << endl;
	}

	fout << "SCALARS fullToCentralDiffFnorm float 1" << endl;
	fout << "LOOKUP_TABLE fullToCentralDiffFnorm_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][60] << endl;
	}

	fout << "SCALARS fewerToCentralDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE fewerToCentralDiffMax_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][61] << endl;
	}

	fout << "SCALARS fewerToCentralDiffFnorm float 1" << endl;
	fout << "LOOKUP_TABLE fewerToCentralDiffFnorm_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][62] << endl;
	}

	fout << "SCALARS fewerToFullDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE fewerToFullDiffMax_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][63] << endl;
	}

	fout << "SCALARS fewerToFullDiffFnorm float 1" << endl;
	fout << "LOOKUP_TABLE fewerToFullDiffFnorm_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][64] << endl;
	}
	
/*	fout << "SCALARS fewerToFullDiffMaxRelative float 1" << endl;
	fout << "LOOKUP_TABLE fewerToFullDiffMaxRelative_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][65] << endl;
	}*/

	fout << "SCALARS virtualToFullDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE virtualToFullDiffMax_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][66] << endl;
	}

	fout << "SCALARS virtualToFullDiffFNorm float 1" << endl;
	fout << "LOOKUP_TABLE virtualToFullDiffFNorm_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][67] << endl;
	}

	fout << "SCALARS virtualToFullDiffDeter float 1" << endl;
	fout << "LOOKUP_TABLE virtualToFullDiffDeter_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][68] << endl;
	}

	fout << "SCALARS virtualToCentralDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE virtualToCentralDiffMax_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][69] << endl;
	}

	fout << "SCALARS virtualToCentralDiffFnorm float 1" << endl;
	fout << "LOOKUP_TABLE virtualToCentralDiffFnorm_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][70] << endl;
	}

	fout << "SCALARS virtualToCentralDiffDeter float 1" << endl;
	fout << "LOOKUP_TABLE virtualToCentralDiffDeter_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[j][71] << endl;
	}

	/*fout << "SCALARS JacobianDiffMaxRelative float 1" << endl;
	fout << "LOOKUP_TABLE JacobianDiffMaxRelative_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[frame-1][j][44] << endl;
	}

	fout << "TENSORS MLSF_Jacobian float" << endl;
	for (int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[frame-1][k];
		fout << temp[10] << " " << temp[11] << " " << temp[12] << endl
		     << temp[13] << " " << temp[14] << " " << temp[15] << endl
		     << temp[16] << " " << temp[17] << " " << temp[18] << endl;
		fout << endl;
	}

	fout << "TENSORS real_Jacobian float" << endl;
	for (int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[frame-1][k];
		fout << temp[19] << " " << temp[20] << " " << temp[21] << endl
		     << temp[22] << " " << temp[23] << " " << temp[24] << endl
		     << temp[25] << " " << temp[26] << " " << temp[27] << endl;
		fout << endl;
	}

	fout << "TENSORS fewer_Jacobian float" << endl;
	for (int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[frame-1][k];
		fout << temp[32] << " " << temp[33] << " " << temp[34] << endl
		     << temp[35] << " " << temp[36] << " " << temp[37] << endl
		     << temp[38] << " " << temp[39] << " " << temp[40] << endl;
		fout << endl;
	}*/

	fout.close();
}


void FTLE_Calculator::generate_interior_vtk(float **data_information, const int& frame, const int& PARTICLE)
{
	std::ofstream fout;
	string frame_str;
	stringstream ss;
	ss << frame;
	ss >> frame_str;
	ss.clear();

	fout.open( ("frame " + frame_str + " interior.vtk").c_str(), ios::out);

	fout << "# vtk DataFile Version 3.0" << endl;
	fout << "Volume example" << endl;
	fout << "ASCII" << endl;
	fout << "DATASET STRUCTURED_POINTS" << endl;
	fout << "DIMENSIONS " << (UNIT-1) << " " << (UNIT-1) << " " << (UNIT-1) << endl;
	fout << "ASPECT_RATIO " << stepSize << " " << stepSize << " " << stepSize << endl;
	fout << "ORIGIN " << stepSize << " " << stepSize << " " << stepSize << endl; 

	fout << "POINT_DATA " << (UNIT-1)*(UNIT-1)*(UNIT-1) << endl;

	int index;
	fout << "SCALARS Radius float 1" << endl;
	fout << "LOOKUP_TABLE radius_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][6] << endl;
			}
		}
	}

	fout << "SCALARS fitted_ftle float 1" << endl;
	fout << "LOOKUP_TABLE mlsf_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][7] << endl;
			}
		}
	}

	fout << "SCALARS RealFtle float 1" << endl;
	fout << "LOOKUP_TABLE real_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][8] << endl;
			}
		}
	}

	fout << "SCALARS ftleDifference float 1" << endl;
	fout << "LOOKUP_TABLE diff_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << (data_information[index][7]-data_information[index][58]) << endl;
			}
		}
	}

	fout << "SCALARS JacobianDiff_Determinant float 1" << endl;
	fout << "LOOKUP_TABLE JacobiandiffDeterminant_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][28] << endl;
			}
		}
	}

	fout << "SCALARS JacobianDifferenceFrobinus float 1" << endl;
	fout << "LOOKUP_TABLE JacobianDiffFrobius_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][29] << endl;
			}
		}
	}

	fout << "SCALARS JacobianDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE JacobianDiffMax_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][30] << endl;
			}
		}
	}

	fout << "SCALARS realJacobianVecError float 1" << endl;
	fout << "LOOKUP_TABLE realJacobianVecError_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][31] << endl;
			}
		}
	}

	fout << "VECTORS velocityDirection float" << endl;
	float *temp = NULL;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				temp = data_information[index];
				fout << temp[3] << " " << temp[4] << " " << temp[5] << endl;
			}
		}
	}

	fout << "SCALARS speed float 1" << endl;
	fout << "LOOKUP_TABLE speed_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << getNorm(&data_information[index][3]) << endl;
			}
		}
	}

	fout << "SCALARS fewerDiff_Determinant float 1" << endl;
	fout << "LOOKUP_TABLE fewerDiffDeterminant_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][41] << endl;
			}
		}
	}

	fout << "SCALARS fewerDifferenceFrobinus float 1" << endl;
	fout << "LOOKUP_TABLE fewerDiffFrobius_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][42] << endl;
			}
		}
	}

	fout << "SCALARS fewerDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE fewerDiffMax_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][43] << endl;
			}
		}
	}

	fout << "SCALARS fittedJacobianVecError float 1" << endl;
	fout << "LOOKUP_TABLE fittedJacobianVecError_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][44] << endl;
			}
		}
	}

	fout << "SCALARS fewerJacobianVecError float 1" << endl;
	fout << "LOOKUP_TABLE fewerJacobianVecError_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][45] << endl;
			}
		}
	}

	fout << "SCALARS centralFTLE float 1" << endl;
	fout << "LOOKUP_TABLE centralFTLE_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][58] << endl;
			}
		}
	}

	fout << "SCALARS fullToCentralMax float 1" << endl;
	fout << "LOOKUP_TABLE fullToCentralMax_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][59] << endl;
			}
		}
	}

	fout << "SCALARS fullToCentralFNorm float 1" << endl;
	fout << "LOOKUP_TABLE fullToCentralFNorm_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][60] << endl;
			}
		}
	}

	fout << "SCALARS fewerToCentralMax float 1" << endl;
	fout << "LOOKUP_TABLE fewerToCentralMax_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][61] << endl;
			}
		}
	}

	fout << "SCALARS fewerToCentralFNorm float 1" << endl;
	fout << "LOOKUP_TABLE fewerToCentralFNorm_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][62] << endl;
			}
		}
	}

	fout << "SCALARS fewerToFullDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE fewerToFullDiffMax_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][63] << endl;
			}
		}
	}

	fout << "SCALARS fewerToFullDiffFNorm float 1" << endl;
	fout << "LOOKUP_TABLE fewerToFullDiffFNorm_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][64] << endl;
			}
		}
	}

/*	fout << "SCALARS fewerToFullDiffMaxRelative float 1" << endl;
	fout << "LOOKUP_TABLE fewerToFullDiffMaxRelative_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][65] << endl;
			}
		}
	}*/

	fout << "SCALARS virtualToFullDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE virtualToFullDiffMax_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][66] << endl;
			}
		}
	}

	fout << "SCALARS virtualToFullDiffFNorm float 1" << endl;
	fout << "LOOKUP_TABLE virtualToFullDiffFNorm_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][67] << endl;
			}
		}
	}

	fout << "SCALARS virtualToFullDiffDeter float 1" << endl;
	fout << "LOOKUP_TABLE virtualToFullDiffDeter_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][68] << endl;
			}
		}
	}

	fout << "SCALARS virtualToCentralDiffMax float 1" << endl;
	fout << "LOOKUP_TABLE virtualToCentralDiffMax_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][69] << endl;
			}
		}
	}

	fout << "SCALARS virtualToCentralDiffFNorm float 1" << endl;
	fout << "LOOKUP_TABLE virtualToCentralDiffFNorm_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][70] << endl;
			}
		}
	}

	fout << "SCALARS virtualToCentralDiffDeter float 1" << endl;
	fout << "LOOKUP_TABLE virtualToCentralDiffDeter_table" << endl;
	for ( int i = 1; i < UNIT; i++)
	{
		for (int j = 1; j < UNIT; ++j)
		{
			for (int k = 1; k < UNIT; ++k)
			{
				index = i*(N)*(N)+j*(N)+k;
				fout << data_information[index][71] << endl;
			}
		}
	}

	/*fout << "SCALARS JacobianDiffMaxRelative float 1" << endl;
	fout << "LOOKUP_TABLE JacobianDiffMaxRelative_table" << endl;
	for ( int j = 0; j < PARTICLE; j++)
	{
		fout << data_information[frame-1][j][44] << endl;
	}

	fout << "TENSORS MLSF_Jacobian float" << endl;
	for (int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[frame-1][k];
		fout << temp[10] << " " << temp[11] << " " << temp[12] << endl
		     << temp[13] << " " << temp[14] << " " << temp[15] << endl
		     << temp[16] << " " << temp[17] << " " << temp[18] << endl;
		fout << endl;
	}

	fout << "TENSORS real_Jacobian float" << endl;
	for (int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[frame-1][k];
		fout << temp[19] << " " << temp[20] << " " << temp[21] << endl
		     << temp[22] << " " << temp[23] << " " << temp[24] << endl
		     << temp[25] << " " << temp[26] << " " << temp[27] << endl;
		fout << endl;
	}

	fout << "TENSORS fewer_Jacobian float" << endl;
	for (int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[frame-1][k];
		fout << temp[32] << " " << temp[33] << " " << temp[34] << endl
		     << temp[35] << " " << temp[36] << " " << temp[37] << endl
		     << temp[38] << " " << temp[39] << " " << temp[40] << endl;
		fout << endl;
	}*/

	fout.close();
}

void FTLE_Calculator::getEachRealJacobian(float *position, const int& frame, const float& time_step, const int& choice)
{
	const float& t = frame*time_step;
	const float& A = ABC_flow::A;
	const float& B = ABC_flow::B;
	const float& C = ABC_flow::C;
	const float& x = position[0];
	const float& y = position[1];
	const float& z = position[2];
	float coefficient;
	switch(choice)
	{
	default:
	case 0:
		{
			coefficient = A+(1-exp(-0.1*t))*sin(2.0*PI*t);
			position[19] = 0;
			position[20] = -C*sin(y);
			position[21] = -coefficient*cos(z);
			position[22] = B*cos(x);
			position[23] = 0;
			position[24] = -coefficient*sin(z);
			position[25] = -B*sin(x);
			position[26] = C*cos(y);
			position[27] = 0;
		}
		break;

	case 1:
		{
			coefficient = A+0.5*t*sin(PI*t);
		    position[19] = 0;
			position[20] = -B*sin(y);
			position[21] = coefficient*cos(z);
			position[22] = B*cos(x);
			position[23] = 0;
			position[24] = -C*sin(z);
			position[25] = -coefficient*sin(x);
			position[26] = C*cos(y);
			position[27] = 0;
		}
		break;

	case 2:
		{
			coefficient = (A+0.5*t*(sin(PI*t)));
			position[19] = 0;
			position[20] = -C*sin(y);
			position[21] = coefficient*cos(z);
			position[22] = B*cos(x);
			position[23] = 0;
			position[24] = -coefficient*sin(z);
			position[25] = -B*sin(x);
			position[26] = C*cos(y);
			position[27] = 0;
		}
		break;

	case 3:
		{
			position[19] = 0;
			position[20] = -C*sin(y);
			position[21] = A*cos(z);
			position[22] = B*cos(x);
			position[23] = 0;
			position[24] = -A*sin(z);
			position[25] = -B*sin(x);
			position[26] = C*cos(y);
			position[27] = 0;
		}
		break;
	}
}


void FTLE_Calculator::getRealFTLE(float **data_information, const int& frame, const int& index, 
								  const int& T, const float& time_step, const int& choice)
{
	float *temp = data_information[index];

	const float& delt_t = T*time_step;
	const float& t1 = frame*time_step;
	const float& t2 = (frame+T)*time_step;

	const float& A = ABC_flow::A;
	const float& B = ABC_flow::B;
	const float& C = ABC_flow::C;
	const float& x = temp[0];
	const float& y = temp[1];
	const float& z = temp[2];
	Matrix3f solution_;
	float right, left;
	switch(choice)
	{
	default:
	case 0:
		{
			left = (exp(-t1/10.0)*(sin(2.0*PI*t1)/10 + 2.0*PI*cos(2.0*PI*t1)))/(4.0*PI*PI + 1.0/100.0);
			right = (exp(-t2/10.0)*(sin(2.0*PI*t2)/10 + 2.0*PI*cos(2.0*PI*t2)))/(4.0*PI*PI + 1.0/100.0);
			solution_(0,0) = 0;
			solution_(0,1) = -C*delt_t*sin(y);
			solution_(0,2) = A*delt_t*cos(z)+(sin(2.0*PI*t2)-sin(2.0*PI*t1))*cos(z)+(right-left)*cos(z);
			solution_(1,0) = B*delt_t*cos(x);
			solution_(1,1) = 0;
			solution_(1,2) = -A*delt_t*sin(z)-(sin(2.0*PI*t2)-sin(2.0*PI*t1))*sin(z)+(right-left)*sin(z);
			solution_(2,0) = -B*delt_t*sin(x);
			solution_(2,1) = C*delt_t*cos(y);
			solution_(2,2) = 0;
		}
		break;

	case 1:
		{
			left = (sin(PI*t1)-PI*t1*cos(PI*t1))/2.0/PI/PI;
			right = (sin(PI*t2)-PI*t2*cos(PI*t2))/2.0/PI/PI;
		    solution_(0,0) = 0;
			solution_(0,1) = -B*delt_t*sin(y);
			solution_(0,2) = A*delt_t*cos(z)+(right-left)*cos(z);
			solution_(1,0) = B*delt_t*cos(x);
			solution_(1,1) = 0;
			solution_(1,2) = -C*delt_t*sin(z);
			solution_(2,0) = -A*delt_t*sin(x)-(right-left)*sin(x);
			solution_(2,1) = C*delt_t*cos(y);
			solution_(2,2) = 0;
		}
		break;

	case 2:
		{
			left = (sin(PI*t1)-PI*t1*cos(PI*t1))/2.0/PI/PI;
			right = (sin(PI*t2)-PI*t2*cos(PI*t2))/2.0/PI/PI;
			solution_(0,0) = 0;
			solution_(0,1) = -C*delt_t*sin(y);
			solution_(0,2) = A*delt_t*cos(z)+(right-left)*cos(z);
			solution_(1,0) = B*delt_t*cos(x);
			solution_(1,1) = 0;
			solution_(1,2) = -A*delt_t*sin(z)-(right-left)*sin(z);
			solution_(2,0) = -B*delt_t*sin(x);
			solution_(2,1) = C*delt_t*cos(y);
			solution_(2,2) = 0;
		}
		break;

	case 3:
		{
			solution_(0,0) = 0;
			solution_(0,1) = -C*delt_t*sin(y);
			solution_(0,2) = A*delt_t*cos(z);
			solution_(1,0) = B*delt_t*cos(x);
			solution_(1,1) = 0;
			solution_(1,2) = -A*delt_t*sin(z);
			solution_(2,0) = -B*delt_t*sin(x);
			solution_(2,1) = C*delt_t*cos(y);
			solution_(2,2) = 0;
		}
		break;
	}

	Matrix3f solution = solution_.transpose()*solution_;	 
	EigenSolver<Matrix3f> result(solution); //result is the complex vector of eigenvalues for matrix solution
	const float& lambda_1 = result.eigenvalues()[0].real();
	const float& lambda_2 = result.eigenvalues()[1].real();
	const float& lambda_3 = result.eigenvalues()[2].real();
	float max_eigen = lambda_1 > lambda_2?(lambda_1 > lambda_3?lambda_1:lambda_3):(lambda_2>lambda_3?lambda_2:lambda_3);
	const float& ftle = log(sqrt(max_eigen))/T;

#ifdef __APPLE__		
	if(_isnan(ftle))
#elif __linux__
	if(isinf(ftle))
#else
    error "Unkown compiler"
    exit(-1)
#endif
	{
		cout << "Error found in inf value for ftle computation!" << endl;
		exit(-1);
	}
	temp[8] = (abs(ftle) < ERROR)? 0.0: ftle;
}


void FTLE_Calculator::computeEachLSF_ftle(float **data_information, const int& index, const int& T)
{
	float *current, *future;
	const std::vector<int>& neighVec = neighbor[index];
	const int& vecSize = neighVec.size();
	MatrixXf curPosition(vecSize,3);
	MatrixXf futPosition(vecSize,3);
	MatrixXf curVelocity(vecSize,3);

	float *currentTarget = data_information[index];
	float *futureTarget = &data_information[index][46];

	for (int i = 0; i < vecSize; ++i)
	{
		current = data_information[neighVec[i]];
		future = &data_information[neighVec[i]][46];
		for (int j = 0; j < 3; ++j)
		{
			curPosition(i,j) = current[j]-currentTarget[j];
			futPosition(i,j) = future[j]-futureTarget[j];
			curVelocity(i,j) = current[j+3]-currentTarget[j+3];
		}
	}

/*----------------------- Moving Least Square Fitting for FTLE value ------------------------------*/	
	MatrixXf ftleMatrix = curPosition.colPivHouseholderQr().solve(futPosition);
	ftleMatrix.transposeInPlace();
	Matrix3f solution = ftleMatrix.transpose()*ftleMatrix;	
	EigenSolver<Matrix3f> result(solution); 

	const float& lambda_1 = result.eigenvalues()[0].real();
	const float& lambda_2 = result.eigenvalues()[1].real();
	const float& lambda_3 = result.eigenvalues()[2].real();
	const float& max_eigen = lambda_1 > lambda_2?(lambda_1 > lambda_3?lambda_1:lambda_3):(lambda_2>lambda_3?lambda_2:lambda_3);
	const float& ftle = log(sqrt(max_eigen))/T;

#ifdef __APPLE__		
	if(_isnan(ftle))
#elif __linux__
	if(isinf(ftle))
#else
    error "Unkown compiler"
    exit(-1)
#endif
	{
		cout << "Error found in inf value for ftle computation!" << endl;
		exit(-1);
	}
	currentTarget[7] = (abs(ftle) < ERROR)? 0.0: ftle;
	currentTarget[9] = currentTarget[7]-currentTarget[8];

/*------------------------ Moving Least Square Fitting for Jacobian ---------------------------------*/
	MatrixXf fittedFullJacobian = curPosition.colPivHouseholderQr().solve(curVelocity);
	fittedFullJacobian.transposeInPlace();

	Matrix3f realJacobian, centralJacobian;
	realJacobian << currentTarget[19], currentTarget[20], currentTarget[21], currentTarget[22], currentTarget[23],
					currentTarget[24], currentTarget[25], currentTarget[26], currentTarget[27];
	centralJacobian << currentTarget[49], currentTarget[50], currentTarget[51], currentTarget[52], currentTarget[53],
					currentTarget[54], currentTarget[55], currentTarget[56], currentTarget[57];
	Matrix3f fullDiff = fittedFullJacobian-realJacobian;
	Matrix3f fullCentralDiff = fittedFullJacobian-centralJacobian;
	currentTarget[28] = fullDiff.determinant();

	float summation = 0.0;
	float maxElement = 0.0;
	float maxRelative = 0.0;

	float centralFullMax = 0.0;
	float centralFullFNorm = 0.0;
	float entry;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			currentTarget[10+i*3+j] = fittedFullJacobian(i,j);
		}
	}
	currentTarget[29] = fullDiff.squaredNorm();
	currentTarget[30] = fullDiff.maxCoeff();
	currentTarget[59] = fullCentralDiff.maxCoeff();
	currentTarget[60] = fullCentralDiff.squaredNorm();

/*----------------------- Fewer Neighbor for Least Square Fitting Computation -------------------------*/
	MatrixXf fewerPosition(vecSize-2,3);
	MatrixXf fewerVelocity(vecSize-2,3);

	for (int i = 0; i < vecSize-2; ++i)
	{
		current = data_information[neighVec[i]];
		for (int j = 0; j < 3; ++j)
		{
			fewerPosition(i,j) = current[j]-currentTarget[j];
			fewerVelocity(i,j) = current[j+3]-currentTarget[j+3];
		}
	}

	MatrixXf fewerJacobian = fewerPosition.colPivHouseholderQr().solve(fewerVelocity);
	fewerJacobian.transposeInPlace();
	MatrixXf fewerDiff = fewerJacobian-realJacobian;
	currentTarget[41] = fewerDiff.determinant();

	Matrix3f fewerCentralDiff = fewerJacobian-centralJacobian;
	currentTarget[61] = fewerCentralDiff.maxCoeff();
	currentTarget[62] = fewerCentralDiff.squaredNorm();

	Matrix3f fewerToFullDiff = fewerJacobian-fittedFullJacobian;
	currentTarget[63] = fewerToFullDiff.maxCoeff();
	currentTarget[64] = fewerToFullDiff.squaredNorm();

	summation = 0.0;
	maxElement = 0.0;
	maxRelative = 0.0;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			currentTarget[32+3*i+j] = fewerJacobian(i,j);
			maxElement = abs(fewerToFullDiff(i,j))/abs(fittedFullJacobian(i,j));
			if(maxRelative<maxElement)
				maxRelative=maxElement;
		}
	}
	
	currentTarget[65] = maxRelative;

	currentTarget[42] = fewerDiff.squaredNorm();
	currentTarget[43] = fewerDiff.maxCoeff();

	summation = 0.0;
	maxElement = 0.0;
	maxRelative = 0.0;
	for (int i = 0; i < vecSize; ++i)
	{
		Vector3f posiDiff(curPosition(i,0), curPosition(i,1), curPosition(i,2));
		Vector3f obtainedVec = realJacobian*posiDiff;
		summation+=(obtainedVec-Vector3f(curVelocity(i,0),curVelocity(i,1),curVelocity(i,2))).squaredNorm();

		Vector3f fittedVec = fittedFullJacobian*posiDiff;
		maxElement+=(fittedVec-Vector3f(curVelocity(i,0),curVelocity(i,1),curVelocity(i,2))).squaredNorm();

		Vector3f fewerVec = fewerJacobian*posiDiff;
		maxRelative+=(fewerVec-Vector3f(curVelocity(i,0),curVelocity(i,1),curVelocity(i,2))).squaredNorm();
	}

	currentTarget[31]=summation;
	currentTarget[44]=maxElement;
	currentTarget[45]=maxRelative;

/********** Use virtual particles in two missing position ********************/

	std::vector<int> neighborTemp;
	float tempVec[3], guassSum, gaussCoeff, *virtualPos, *virtualNeigh;
	int virtualIndex;
	for (int i = 0; i < 2; ++i)
	{
		float radius = 1.0/50.0*PI*INITIAL;
		virtualIndex = vecSize-2+i;
		virtualPos = &(data_information[neighVec[virtualIndex]][0]);
		find_neigbhor_index(data_information, virtualPos, radius, neighborTemp);
		memset(tempVec,0,3*sizeof(float));
		guassSum = 0.0;
		for (int j = 0; j < neighborTemp.size(); ++j)
		{
			virtualNeigh = &(data_information[neighborTemp[j]][0]);
			gaussCoeff = getGaussian(virtualPos, virtualNeigh, radius);
			guassSum += gaussCoeff;
			tempVec[0]+=gaussCoeff*virtualNeigh[3];
			tempVec[1]+=gaussCoeff*virtualNeigh[4];
			tempVec[2]+=gaussCoeff*virtualNeigh[5];
		}
		if(guassSum==0.0)
		{
			std::cout << "Error for Gaussian summation!" << std::endl;
			exit(1);
		}
		tempVec[0]/=guassSum, tempVec[1]/=guassSum, tempVec[2]/=guassSum;

		curVelocity(virtualIndex,0) = tempVec[0]-currentTarget[3];
		curVelocity(virtualIndex,1) = tempVec[1]-currentTarget[4];
		curVelocity(virtualIndex,2) = tempVec[2]-currentTarget[5];
		neighborTemp.clear();
	}
	MatrixXf virtualPosJacobian = curPosition.colPivHouseholderQr().solve(curVelocity);
	virtualPosJacobian.transposeInPlace();

	MatrixXf virtual_to_central_diff = virtualPosJacobian-centralJacobian;
	MatrixXf virtual_to_full_diff = virtualPosJacobian-fittedFullJacobian;

	currentTarget[66] = virtual_to_full_diff.maxCoeff();
	currentTarget[67] = virtual_to_full_diff.squaredNorm();
	currentTarget[69] = virtual_to_central_diff.maxCoeff();
	currentTarget[70] = virtual_to_central_diff.squaredNorm();

	/*maxRelative = 0.0;
	summation = 0.0;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			maxElement = abs(virtual_to_full_diff(i,j))/abs(fittedFullJacobian(i,j));
			if(maxRelative<maxElement)
				maxRelative=maxElement;

			maxElement = abs(virtual_to_central_diff(i,j))/abs(centralJacobian(i,j));
			if(summation<maxElement)
				summation=maxElement;
		}
	}*/
	currentTarget[68] = virtual_to_full_diff.determinant();
	currentTarget[71] = virtual_to_central_diff.determinant();

}


void FTLE_Calculator::get_limit(float **data_information, const int& PARTICLE, float readme[][2])
{
	float *temp = NULL;
	for (int i = 0; i < PARTICLE; ++i)
	{
		temp = data_information[i];
		if(readme[0][0]>temp[6])
			readme[0][0]=temp[6];
		if(readme[0][1]<temp[6])
			readme[0][1]=temp[6];
		if(readme[1][0]>temp[7])
			readme[1][0]=temp[7];
		if(readme[1][1]<temp[7])
			readme[1][1]=temp[7];
		if(readme[2][0]>temp[8])
			readme[2][0]=temp[8];
		if(readme[2][1]<temp[8])
			readme[2][1]=temp[8];
		if(readme[3][0]>temp[9])
			readme[3][0]=temp[9];
		if(readme[3][1]<temp[9])
			readme[3][1]=temp[9];
	}
}


const float getGaussian(float *virtualPos, float *virtualNeigh, const float &radius)
{
	return exp(-((virtualPos[0]-virtualNeigh[0])*(virtualPos[0]-virtualNeigh[0])
		      +(virtualPos[1]-virtualNeigh[1])*(virtualPos[1]-virtualNeigh[1])
		      +(virtualPos[2]-virtualNeigh[2])*(virtualPos[2]-virtualNeigh[2]))
			   /2.0/radius/radius);
}