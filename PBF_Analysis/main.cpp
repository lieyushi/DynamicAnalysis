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
const int FRAME = 1499;
const int PARTICLE = 128000;
const float ERROR = 1.0e-8;
const float Stiffness = 0.01;

const int NUMBER = 100;
float **data_information[NUMBER]; //this will store information from files

/*
		0 1 2		position
		3 4 5		velocity
		6			density
		7			neighbor_number
		8 9 10		velocity_norm_gradient
		11 12 13	rotation_gradient
		14			ftle
		15			rotation
		16 -- 24	symmetric Jacobian
		25 -- 33	antisymmetric Jacobian
		34 -- 42	Jacobian
		43			Qvalue
		44			lambda_2
		45			divergence
		46			curl
		47			Jacobian-Det
		48			Jacobian_FNorm
		49			DorminantMap
		50			radius
		51 			octonary_difference
		52			octonaryVarience
		53			gasEnergy
		54			Tait_energy
		55			vecErrorNorm
*/


vector< vector<int > > neighbor(PARTICLE); //this will store neighbor of the current frame
vector< Hashing::Block> block_list;
const int& T = 50;
const float& RADIUS = 2;
Hashing::Parameter para;

const float Boundary[3][2] = {-2.0, 48,
							  -2.0, 120,
							  -2.0, 128};
/* setting the file absolute path */
const string& location = "/media/lieyu/Seagate Backup Plus Drive/PBF_2013Macklin/pbf_velocitySeparate/source_data/";

const float& TIMES = 1.1;
const int& THRESHOLD = 12;
const int& VOLME_NEIGHBOR = 5;
const float& SIGMA = 8.0;
const float& restDensity = 1000.0;
const float& Bottom = Boundary[1][0];
const float& Gravity = 9.81;
const float& Phi = 7.0;


const float distance(float *a, float *b);
const float Gaussian_weight(float *position, float *x, const float& sigma);
const vector<int> find_neigbhor_volume(const int& frame, float *position, float& radius);
const float norm(float *a);
void initialize_memory_file_reader();
void release_memory_file_reader();
void load_data(const int& start);
void find_neigbhor_index(const int& frame, const int& number, float& radius);
void find_neighbor(const int& frame, const float& radius);
void my_init();
void assigned_grid(const int& frame, const float& radius);
void compute_FTLE(const int& frame, const float& B, const bool& isWeighted);
void generate_txt(const int& row, const int& frame, float extrema[]);
void volume_rendering(const int& row, const int& frame,const float& radius, const bool& normalized);  
void generate_vtk_particles(const int& row, const int& frame); 
void ComputeRotation(const int& frame, const int& i, const float& B);
void ComputeGradient(const int& frame, const int& particle, const bool& isWeighted);
void computeRotationGradient(const int& frame, const int& particle);


float extrema[16] = {FLT_MAX, FLT_MIN, /* density */
					  FLT_MAX, FLT_MIN, /* velocity */
					  FLT_MAX, FLT_MIN, /* ftle */
					  FLT_MAX, FLT_MIN, /* rotation */
					  FLT_MAX, FLT_MIN, /* Qvalue */
					  FLT_MAX, FLT_MIN, /* lambda2 */
					  FLT_MAX, FLT_MIN, /* divergence */
					  FLT_MAX, FLT_MIN /* curl */
                      //INT_MAX, INT_MIN, /* meanError */
                      //INT_MAX, INT_MIN
                      }; /* maxError */


int main(int argc, char **argv)
{
	if(argc != 2 && argc != 3)
	{
		perror("Incomplete arguments! Should have ./main 0(notNormalized)/1(Normalized) number!\n");
		exit(-1);
	}
	bool normalized;
	if(atoi(argv[1]) == 0)
		normalized = false;
	else if(atoi(argv[1]) == 1)
		normalized = true;

	int weighted;
	cout << "-------- Choose whether use weighted least square or not? 0)No, 1)Yes!----------------" << endl;
	cout << "Choice: ";
	cin >> weighted;
	if(weighted!=0 && weighted!=1)
	{
		cout << "Error read input for weighted choice!" << endl;
		exit(1);
	}
	cout << endl;
	bool isWeighted;
	if(weighted==0)
		isWeighted = false;
	else if(weighted==1)
		isWeighted = true;

	float io_time = 0;
	clock_t starting, ending;
	time(&starting);
	initialize_memory_file_reader();

	const float& B = 2.0*Gravity*40*0.05*100/Phi;
	
	if(argc == 2)
	{
		for (int i = 0; i < ceil((float)(FRAME-T)/(float)(NUMBER-T)); i++)
		{
			clock_t begin, end;
			time(&begin);
			
			load_data((NUMBER - T)*i+1);
			time(&end);
			io_time += (float)difftime(end, begin);

			for (int j = 0; j < NUMBER - T; j++)//compute at most NUMBER-T+1 FTLE values
			{
				const int& frame = j + i*(NUMBER - T); //frame will be from 1 -> 1000 - 5
				if (frame > FRAME - T)
				{
					break;
				}
				cout << "Frame " << frame << " computation for FTLE starts!" << endl;

				assigned_grid(j, RADIUS);			
				find_neighbor(j, RADIUS);
				compute_FTLE(j, B, isWeighted);

				generate_txt(frame, j, extrema);

				volume_rendering(frame, j, RADIUS, normalized);
				block_list.clear();
				generate_vtk_particles(frame, j);
				
				cout << "Frame " << frame << " computation ends!" << endl;
			}
		}	
	}
	else if(argc==3)
	{
		const int& fileStart = atoi(argv[2]);
		clock_t begin, end;
		time(&begin);
		
		load_data(fileStart);
		time(&end);
		io_time += (float)difftime(end, begin);

		for (int j = 0; j < NUMBER - T; j++)//compute at most NUMBER-T+1 FTLE values
		{
			const int& frame = j + fileStart; //frame will be from 1 -> 1000 - 5
			if (frame > FRAME - T)
			{
				break;
			}
			cout << "Frame " << frame << " computation for FTLE starts!" << endl;

			assigned_grid(j, RADIUS);			
			find_neighbor(j, RADIUS);
			compute_FTLE(j, B, isWeighted);

			generate_txt(frame, j, extrema);

			volume_rendering(frame, j, RADIUS, normalized);
			block_list.clear();
			generate_vtk_particles(frame, j);
			
			cout << "Frame " << frame << " computation ends!" << endl;
		}

	}
	release_memory_file_reader();
	time(&ending);

	ofstream out("README", ios::out);
	out << "Total computational time is: " << float(difftime(ending, starting)) << " S!" << endl;
	out << "I-O time is: " << io_time << " S" << endl;
	out << "Density range is: " << extrema[0] << ", " << extrema[1] << endl;
	out << "Velocity value is: " << extrema[2] << ", " << extrema[3] << endl;
	out << "FTLE value range is: " << extrema[4] << ", " << extrema[5] << endl;
	out << "Rotation value range is: " << extrema[6] << ", " << extrema[7] << endl;
	out << "Qvalue range is: " << extrema[8] << ", " << extrema[9] << endl;
	out << "Lambda2 value is: " << extrema[10] << ", " << extrema[11] << endl;
	out << "Divergence value is: " << extrema[12] << ", " << extrema[13] << endl;
	out << "Curl value is: " << extrema[14] << ", " << extrema[15] << endl;
	//out << "MeanError value is: " << extrema[16] << ", " << extrema[17] << endl;
	//out << "MaxError value is: " << extrema[18] << ", " << extrema[19] << endl;
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

	float *temp;

	fout << "# vtk DataFile Version 3.0" << endl;
	fout << "PBF_FTLE example" << endl;
	fout << "ASCII" << endl;
	fout << "DATASET UNSTRUCTURED_GRID" << endl;
	fout << "POINTS " << PARTICLE << " float" << endl;
	for( int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[j][k];
		fout << temp[0] << " " << temp[1] << " " << temp[2] << endl; 	 
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

	fout << "VECTORS velocityDirection float" << endl;
    for ( int k = 0; k < PARTICLE; k++)
	{
		temp = data_information[j][k];
		fout << temp[3] << " " << temp[4] << " " << temp[5] << endl;
	}

	fout << "SCALARS ftle_neighbor int 1" << endl;
	fout << "LOOKUP_TABLE neighbortable" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][7] << endl;
	}	


	fout << "SCALARS speed float 1" << endl;
	fout << "LOOKUP_TABLE speed_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		temp = data_information[j][k];
		fout << norm(&(temp[3])) << endl;
	}


	fout << "SCALARS rotationGrad float 1" << endl;
	fout << "LOOKUP_TABLE rotationGrad_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		temp = data_information[j][k];
		fout << norm(&(temp[11])) << endl;
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

	fout << "SCALARS q_value float 1" << endl;
	fout << "LOOKUP_TABLE qvalue_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][43] << endl;
	}

	fout << "SCALARS lamba_2 float 1" << endl;
	fout << "LOOKUP_TABLE lamba_2_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][44] << endl;
	}

	fout << "SCALARS divergence float 1" << endl;
	fout << "LOOKUP_TABLE divergence_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][45] << endl;
	}

	fout << "SCALARS curl float 1" << endl;
	fout << "LOOKUP_TABLE curl_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][46] << endl;
	}

	fout << "SCALARS JacobianPositive int 1" << endl;
	fout << "LOOKUP_TABLE JacobianPositive_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][47] << endl;
	}

	fout << "SCALARS JacobianFrobinus float 1" << endl;
	fout << "LOOKUP_TABLE Frobinus_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][48] << endl;
	}

	fout << "SCALARS Dorminant int 1" << endl;
	fout << "LOOKUP_TABLE dorminance_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][49] << endl;
	}

	fout << "SCALARS searchingRadius float 1" << endl;
	fout << "LOOKUP_TABLE radius_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][50] << endl;
	}	

	fout << "SCALARS octonaryDiff float 1" << endl;
	fout << "LOOKUP_TABLE octonary_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][51] << endl;
	}	

	fout << "SCALARS octonaryVarience float 1" << endl;
	fout << "LOOKUP_TABLE octonaryVar_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][52] << endl;
	}

	fout << "SCALARS gasEnergy float 1" << endl;
	fout << "LOOKUP_TABLE gasEnergy_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][53] << endl;
	}

	fout << "SCALARS TaitEnergy float 1" << endl;
	fout << "LOOKUP_TABLE TaitEnergy_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][54] << endl;
	}

	fout << "SCALARS vecErrorNorm float 1" << endl;
	fout << "LOOKUP_TABLE vecErrorNorm_table" << endl;
	for ( int k = 0; k < PARTICLE; k++)
	{
		fout << data_information[j][k][55] << endl;
	}


	fout << "TENSORS symmetric float" << endl;
	for (int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[j][k];
		fout << temp[16] << " " << temp[17] << " " << temp[18] << endl
		     << temp[19] << " " << temp[20] << " " << temp[21] << endl
		     << temp[22] << " " << temp[23] << " " << temp[24] << endl;
		fout << endl;
	}

	fout << "TENSORS antisymmetric float" << endl;
	for (int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[j][k];
		fout << temp[25] << " " << temp[26] << " " << temp[27] << endl
		     << temp[28] << " " << temp[29] << " " << temp[30] << endl
		     << temp[31] << " " << temp[32] << " " << temp[33] << endl;
		fout << endl;
	}

	fout << "TENSORS Jacobian float" << endl;
	for (int k = 0; k < PARTICLE; ++k)
	{
		temp = data_information[j][k];
		fout << temp[34] << " " << temp[35] << " " << temp[36] << endl
		     << temp[37] << " " << temp[38] << " " << temp[39] << endl
		     << temp[40] << " " << temp[41] << " " << temp[42] << endl;
		fout << endl;
	}
	fout.close(); 
}



const float Gaussian_weight(float *position, float *x, const float& sigma)
{
	return exp(-pow(distance(position,x),2.0)/2.0/pow(sigma,2.0));
}



const float norm(float *a)
{
	return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}



void volume_rendering(const int& j, const int& frame, const float& radius, const bool& normalized)
{
    const float& interval = radius/2.0;

/*distribution  order of particles on the surface, firstly x, then y, then z */
	const int& XSize = (Boundary[0][1]-Boundary[0][0])/interval + 1;
	const int& YSize = (Boundary[1][1]-Boundary[1][0])/interval + 1;
	const int& ZSize = (Boundary[2][1]-Boundary[2][0])/interval + 1;
	const int& total = XSize*YSize*ZSize;
	float **scalar = new float*[total];

	if(!normalized) /* not normalized, but directly computed from Gaussian interpolation summation */
	{

	#pragma omp parallel for num_threads(8)
		for(int i = 0; i < total; i++)
		{

			scalar[i] = new float[17];
			memset(scalar[i],0,sizeof(float)*18);

			/* define an initialize value for gaussian search kernel */
			float volume_radius = radius;
			float position[3] = {i%XSize*interval + para.RANGE[0][0],
										i/XSize%YSize*interval + para.RANGE[1][0],
										i/XSize/YSize*interval + para.RANGE[2][0]};
			const vector<int>& support = find_neigbhor_volume(frame, position, volume_radius);
			if(support.size() == 0)
				continue;

			for(int j = 0; j < support.size(); j++)
			{
				float *temp = data_information[frame][support[j]];
				float x[3] = {temp[0], temp[1], temp[2]};

				const float& gaussion = Gaussian_weight(position, x, radius/SIGMA);
				//dorminant += gaussion;
				scalar[i][0] += temp[6] * gaussion;
				scalar[i][1] += norm(&(temp[8])) * gaussion;
				scalar[i][2] += temp[14] * gaussion;
				scalar[i][3] += temp[15] * gaussion;
				scalar[i][4] += temp[43] * gaussion;
				scalar[i][5] += temp[44] * gaussion;
				scalar[i][6] += temp[45] * gaussion;
				scalar[i][7] += temp[46] * gaussion;
				scalar[i][8] += temp[3] * gaussion;
				scalar[i][9] += temp[4] * gaussion;
				scalar[i][10] += temp[5] * gaussion;
				scalar[i][11] += norm(&(temp[3])) * gaussion;
				scalar[i][12] += temp[51] * gaussion;
				scalar[i][13] += temp[52] * gaussion;
				scalar[i][14] += temp[53] * gaussion;
				scalar[i][15] += temp[54] * gaussion;
				scalar[i][16] += temp[55] * gaussion;
				scalar[i][17] += norm(&(temp[11])) * gaussion;
			}

		}
	}
	else
	{
	#pragma omp parallel for num_threads(8)
		for(int i = 0; i < total; i++)
		{

			scalar[i] = new float[18];
			memset(scalar[i],0,sizeof(float)*18);

			/* define an initialize value for gaussian search kernel */
			float volume_radius = radius;
			float position[3] = {i%XSize*interval + para.RANGE[0][0],
										i/XSize%YSize*interval + para.RANGE[1][0],
										i/XSize/YSize*interval + para.RANGE[2][0]};
			const vector<int>& support = find_neigbhor_volume(frame, position, volume_radius);
			if(support.size() == 0)
				continue;

			/* if not normalized required, then no need to define a dornominant */
			float dorminant = 0.0;
			for(int j = 0; j < support.size(); j++)
			{
				float *temp = data_information[frame][support[j]];
				float x[3] = {temp[0], temp[1], temp[2]};

				const float& gaussion = Gaussian_weight(position, x, radius/SIGMA);
				dorminant += gaussion;
				scalar[i][0] += temp[6] * gaussion;
				scalar[i][1] += norm(&(temp[8])) * gaussion;
				scalar[i][2] += temp[14] * gaussion;
				scalar[i][3] += temp[15] * gaussion;
				scalar[i][4] += temp[43] * gaussion;
				scalar[i][5] += temp[44] * gaussion;
				scalar[i][6] += temp[45] * gaussion;
				scalar[i][7] += temp[46] * gaussion;
				scalar[i][8] += temp[3] * gaussion;
				scalar[i][9] += temp[4] * gaussion;
				scalar[i][10] += temp[5] * gaussion;
				scalar[i][11] += norm(&(temp[3])) * gaussion;
				scalar[i][12] = temp[51] * gaussion;
				scalar[i][13] = temp[52] * gaussion;
				scalar[i][14] += temp[53] * gaussion;
				scalar[i][15] += temp[54] * gaussion;
				scalar[i][16] += temp[55] * gaussion;
				scalar[i][17] += norm(&(temp[11])) * gaussion;
			}
			if(dorminant == 0.0)
			{
				cout << "Dorminant found to be zero!" << endl;
				cout << dorminant << endl;
				exit(-1);
			}
			for (int j = 0; j < 18; ++i)
			{
				scalar[i][j]/=dorminant;
			}
		}
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

	fout << "SCALARS speedGradient float 1" << endl;
	fout << "LOOKUP_TABLE dv_table" << endl;
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

	fout << "SCALARS Qvalue float 1" << endl;
	fout << "LOOKUP_TABLE qvalue_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][4] << endl;
	}

	fout << "SCALARS lambda2 float 1" << endl;
	fout << "LOOKUP_TABLE lambda2_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][5] << endl;
	}

	fout << "SCALARS divergence float 1" << endl;
	fout << "LOOKUP_TABLE divergence_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][6] << endl;
	}

	fout << "SCALARS curl float 1" << endl;
	fout << "LOOKUP_TABLE curl_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][7] << endl;
	}

	fout << "SCALARS speed float 1" << endl;
	fout << "LOOKUP_TABLE speed_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][11] << endl;
	}

	fout << "SCALARS octonaryDiff float 1" << endl;
	fout << "LOOKUP_TABLE octoDif_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][12] << endl;
	}

	fout << "SCALARS octonaryVar float 1" << endl;
	fout << "LOOKUP_TABLE octoVar_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][13] << endl;
	}

	fout << "SCALARS gasEnergy float 1" << endl;
	fout << "LOOKUP_TABLE gasEnergy_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][14] << endl;
	}

	fout << "SCALARS TaitEnergy float 1" << endl;
	fout << "LOOKUP_TABLE TaitEnergy_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][15] << endl;
	}

	fout << "SCALARS vecNormError float 1" << endl;
	fout << "LOOKUP_TABLE vecNormError_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][16] << endl;
	}

	fout << "SCALARS rotationGrad float 1" << endl;
	fout << "LOOKUP_TABLE rotationGrad_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][17] << endl;
	}

	/*fout << "SCALARS meanError float 1" << endl;
	fout << "LOOKUP_TABLE meanError_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][8] << endl;
	}

	fout << "SCALARS maxError float 1" << endl;
	fout << "LOOKUP_TABLE maxError_table" << endl;
	for(int i = 0; i < total; i++)
	{
		fout << scalar[i][9] << endl;
	}*/

	fout.close();

#pragma omp parallel for num_threads(8)
	for(int i = 0; i < total; i++)
	{
		delete[] scalar[i];
	}
	delete[] scalar;
}


void generate_txt(const int& actual, const int& frame, float extrema[])
{
	string frame_str;
	stringstream ss;
	ss << actual;
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
			"density_gradient(3), ftle(1), rotation(1), symmetric(9), antisymmetric(9), Jacobian(9), Qvalue(1), lambda_2(1), divergence(1)" << 
			 ", curl(1), JacobianDeterminant(1), JacobianFNorm(1), DorminantMap(1) and radius(1)!" << endl;

	float *temp;	

	float GasSummation = 0.0;
	float TaitSummation = 0.0;
	float maxEnergy = INT_MIN;

	for(int i = 0; i < PARTICLE; i++)
	{
		temp = data_information[frame][i];
		GasSummation += temp[53];
		TaitSummation += temp[54];
		if(maxEnergy<temp[54])
			maxEnergy=temp[54];
		for(int j = 0; j < 56; j++)
		{
			fout << temp[j] << " ";
		}
		fout << endl;

		/*if(temp[6] < extrema[0])
			extrema[0] = temp[6];
		if(temp[6] > extrema[1])
			extrema[1] = temp[6];

		float v[3] = {temp[3], temp[4], temp[5]};
		if(norm(v) < extrema[2])
			extrema[2] = norm(v);
		if(norm(v) > extrema[3])
			extrema[3] = norm(v);	
		
		if(temp[14] < extrema[4])
			extrema[4] = temp[14];
		if(temp[14] > extrema[5])
			extrema[5] = temp[14];

		if(temp[15] < extrema[6])
			extrema[6] = temp[15];
		if(temp[15] > extrema[7])
			extrema[7] = temp[15];

		if(temp[43] < extrema[8])
			extrema[8] = temp[43];
		if(temp[43] > extrema[9])
			extrema[9] = temp[43];	

		if(temp[44] < extrema[10])
			extrema[10] = temp[44];
		if(temp[44] > extrema[11])
			extrema[11] = temp[44];

		if(temp[45] < extrema[12])
			extrema[12] = temp[45];
		if(temp[45] > extrema[13])
			extrema[13] = temp[45];	

		if(temp[46] < extrema[14])
			extrema[14] = temp[46];
		if(temp[46] > extrema[15])
			extrema[15] = temp[46];*/
	}
	fout.close();
	cout << "Data written into text file for frame " << frame << endl;

	GasSummation/=(float)PARTICLE;
	TaitSummation/=(float)PARTICLE;
	ofstream energy("Energy.txt", ios::out | ios::app);
	if(!energy)
	{
		cout << "Error creating such file!" << endl;
		exit(-1);
	}
	energy << actual << " " << GasSummation << " " << TaitSummation << " " << maxEnergy << endl;
	energy.close();
}



void initialize_memory_file_reader()
{
#pragma omp parallel for num_threads(8)
	for(int i = 0; i < NUMBER; i++)
	{
		data_information[i] = new float *[PARTICLE];
		for (int j = 0; j < PARTICLE; j++)
		{
			data_information[i][j] = new float[56];
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

	neighbor.clear();
	block_list.clear();
	std::cout << "Memory elimination completed! " << std::endl;
	std::cout << std::endl;
}


void load_data(const int& start) //start from 1
{

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
		float a;
		float *temp;
		for (int j = 0; j < PARTICLE; j++)
		{
			temp = data_information[i-start][j];
			in_ >> a >> temp[0] >> temp[1] >> temp[2]
				>> temp[3] >> temp[4] >> temp[5] 
				>> a >> a >> a >> temp[6];
			//in_ >> b;
		}
		ss.clear();
		in_.close();
		cout << "File " << (i) << " has been loaded successfully!" << endl;
	}

	cout << "Finished I-O Operations!" << endl;
}


void assigned_grid(const int& frame, const float& radius)
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
		float *temp = data_information[frame][i];
		const int& a = floor((temp[0] - para.RANGE[0][0])/para.grid_size[0]);
		const int& b = floor((temp[1] - para.RANGE[1][0])/para.grid_size[1]);
		const int& c = floor((temp[2] - para.RANGE[2][0])/para.grid_size[2]);
		const int& index = a + b*para.grid[0] + c*para.grid[0]*para.grid[1];
		assert(index >= 0 && index < size);
		block_list[index].add(i);
	}
	cout << "Frame " << frame << " assigning particle positions completed!" << endl;
}



void find_neighbor(const int& frame, const float& radius)
{
	cout << "Starting searching neighbor!" << endl;

#pragma omp parallel for num_threads(8)
	for (int i = 0; i < PARTICLE; i++)
	{
		neighbor[i].clear();
		float *temp = data_information[frame][i];
		temp[50] = radius;
		find_neigbhor_index(frame, i, temp[50]);
		temp[7] = neighbor[i].size();
	}
}


const vector<int> find_neigbhor_volume(const int& frame, float *position, float& radius)
{
	vector<int> my_neighbor;
	int search = 0;
	bool flag = false;
	do
	{
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
							float x_[3] = {data_information[frame][block_list[_num].inside[_i]][0], 
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
		if (my_neighbor.size() >= VOLME_NEIGHBOR || search >= 4)
			flag = true;
		else
		{
			radius *= TIMES;
			search++;
		}
	}while(!flag);
	
	return my_neighbor;
}



void find_neigbhor_index(const int& frame, const int& number, float& radius)
{
	bool flag = false;
	/* compute how many blocks should be searched along each dimension */
	float *temp = data_information[frame][number];
	do
	{
		int ratio[3] = {ceil(radius/para.grid_size[0]),
						ceil(radius/para.grid_size[1]),
						ceil(radius/para.grid_size[2])};
		int index[3];

		index[0] = floor((temp[0] - para.RANGE[0][0])/para.grid_size[0]);
		index[1] = floor((temp[1] - para.RANGE[1][0])/para.grid_size[1]);
		index[2] = floor((temp[2] - para.RANGE[2][0])/para.grid_size[2]);
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
		{
			radius *= TIMES;
		}

	}while(!flag);
}



const float distance(float *a, float *b)
{
	return sqrt(pow(a[0]-b[0],2.0) + pow(a[1]-b[1],2.0) + pow(a[2]-b[2],2.0));
}



void compute_FTLE(const int& frame, const float& B, const bool& isWeighted) //frame starts from 1 to FRAME - T
{

#pragma omp parallel for num_threads(8)
	for (int i = 0; i < PARTICLE; i++)
	{
		/* compute rotation along time window size T */
		ComputeRotation(frame, i, B);

		ComputeGradient(frame, i, isWeighted);	
	}

#pragma omp parallel for num_threads(8)
	for (int i = 0; i < PARTICLE; ++i)
		{
			computeRotationGradient(frame, i);
		}	
}


void ComputeRotation(const int& frame, const int& i, const float& B)
{

/*************** Compute rotation field for each particle within the T window size  ********************/
	float result = 0.0, v1_length, v2_length, angle;
	float *first, *second, dot_;
	glm::vec3 v_1, v_2;

	float* target = data_information[frame][i];

	for(int j = 0; j < T; j++)
	{
		first = data_information[frame+j][i];
		second = data_information[frame+j+1][i];

		v_1 = glm::vec3(first[3], first[4], first[5]);
		v_2 = glm::vec3 (second[3], second[4], second[5]);
		v1_length = glm::length(v_1);
		v2_length = glm::length(v_2);
		dot_ = glm::dot(v_1,v_2);
		if(v1_length>1.0e-6&&v2_length>1.0e-6)
		{
			angle = dot_/v1_length/v2_length;
			angle = min(1.0,double(angle));
			angle = max(-1.0,double(angle));
			result += acos(angle);
		}
	}

	target[15] = (abs(result) < ERROR)? 0.0: result;

	float linearPressure, expoPressure;
	if(target[6] <= restDensity)
	{
		linearPressure = 0.0;
		expoPressure = 0.0;
	}
	else
	{
		linearPressure = Stiffness*(target[6]-restDensity);
		expoPressure = B*(pow(target[6]/restDensity,Phi)-1.0);
		linearPressure/=target[6];
		expoPressure/=target[6];
	}
	linearPressure+=0.5*(target[3]*target[3]+target[4]*target[4]+target[5]*target[5]);
	linearPressure+=(target[1]-Bottom)*Gravity;

	expoPressure+=0.5*(target[3]*target[3]+target[4]*target[4]+target[5]*target[5]);
	expoPressure+=(target[1]-Bottom)*Gravity;

	target[53] = linearPressure;

	target[54] = expoPressure;

}



void ComputeGradient(const int& frame, const int& particle, const bool& isWeighted)
{

/*************** Use least square fitting technique to compute ftle and gradient  ********************/

	const int& future =  frame+T;

	const vector<int>& vec = neighbor[particle];

	const int& vecSize = vec.size();

	assert(vecSize!=0);

	MatrixXf previousPosition(vecSize, 3);

	VectorXf velocityNorm(vecSize);

	MatrixXf currentPosition(vecSize, 3);

	MatrixXf velocityVec(vecSize, 3);

	//VectorXf densVariation(vecSize);

	/* current frame information */
	float* prevTarget = data_information[frame][particle];

	const float& veloNorm = sqrt(prevTarget[3]*prevTarget[3]+prevTarget[4]*prevTarget[4]+prevTarget[5]*prevTarget[5]);

	/* future frame information */
	const float* currentTarget = data_information[future][particle];

	float *prev, *current, xDir, yDir, zDir;

	int index;

	glm::vec3 orientation;

	const glm::vec3& xAxis = glm::vec3(1.0, 0.0, 0.0);
	const glm::vec3& yAxis = glm::vec3(0.0, 1.0, 0.0);
	const glm::vec3& zAxis = glm::vec3(0.0, 0.0, 1.0);

	std::vector<int> store(8,0);
	int a,b,c,p;

	std::vector<int> belonging;

	for (int i = 0; i < vecSize; i++)
	{
		index = vec[i];

		prev = data_information[frame][index];

		current = data_information[future][index];

		/* test whether neighbor particles are distributed in octonal area or not */

		orientation = glm::vec3(prev[0]-prevTarget[0], prev[1]-prevTarget[1], prev[2]-prevTarget[2]);

		xDir = glm::dot(orientation, xAxis);
		yDir = glm::dot(orientation, yAxis);
		zDir = glm::dot(orientation, zAxis);

		if(xDir<=0.0)
			a = 0;
		else if(xDir>0.0)
			a = 1;

		if(yDir<=0.0)
			b = 0;
		else if(yDir>0.0)
			b = 1;

		if(zDir<=0.0)
			c = 0;
		else if(zDir>0.0)
			c = 1;

		p = a+b*2+c*4;

		if(isWeighted)
			belonging.push_back(p);

		store[p]++;

		for (int j = 0; j < 3; ++j)
		{
			previousPosition(i,j) = prev[j] - prevTarget[j];
			currentPosition(i,j) = current[j] - currentTarget[j];
			velocityVec(i,j) = prev[3+j] - prevTarget[3+j];
		}

		velocityNorm(i) = sqrt(prev[3]*prev[3]+prev[4]*prev[4]+prev[5]*prev[5]) - veloNorm;
		//densVariation(i) = prev[6]-prevTarget[6];
	}

/**************Compute FTLE from least square fitting from current and future neighbor positions *****************************/
	MatrixXf ftleMatrix = previousPosition.colPivHouseholderQr().solve(currentPosition);
	ftleMatrix.transposeInPlace();
	Matrix3f solution = ftleMatrix.transpose() * ftleMatrix;	
	EigenSolver<Matrix3f> result(solution); 

	const float& lambda_1 = result.eigenvalues()[0].real();
	const float& lambda_2 = result.eigenvalues()[1].real();
	const float& lambda_3 = result.eigenvalues()[2].real();
	const float& max_eigen = lambda_1 > lambda_2?(lambda_1 > lambda_3?lambda_1:lambda_3):(lambda_2>lambda_3?lambda_2:lambda_3);
	const float& ftle = log(sqrt(max_eigen))/T;

	if(isinf(ftle))
	{
		cout << "Found inf value for ftle! Please check source code!" << endl;
		exit(-1);
	}

	prevTarget[14] = (abs(ftle) < ERROR? 0.0: ftle);


/**********************Compute gradient for velocity norm and density **************************************************/
	Vector3f normGradient = previousPosition.colPivHouseholderQr().solve(velocityNorm);
	//Vector3f denGradient = previousPosition.colPivHouseholderQr().solve(densVariation);
	for (int i = 0; i < 3; ++i)
	{
		prevTarget[8+i] = normGradient(i);
		//prevTarget[11+i] = denGradient(i);
	}


/**********************Compute Local Jacobian **************************************************/
	MatrixXf Jacobian;

	if(isWeighted)
	{
		MatrixXf W_matrix = Eigen::MatrixXf(vecSize,vecSize);
		W_matrix.setZero();
		int space;
		for (int i = 0; i < vecSize; ++i)
		{	
			space = store[belonging[i]];
			if(space == 0)
			{
				cout << "Error for counting octonary number!" << endl;
				exit(1);
			}
			W_matrix(i,i) = 1.0/(float)space;
		}
		MatrixXf leftMatrix = previousPosition.transpose()*W_matrix*previousPosition;
		MatrixXf rightMatrix = previousPosition.transpose()*W_matrix*velocityVec;
		Jacobian = leftMatrix.colPivHouseholderQr().solve(rightMatrix);
	}
	else
		Jacobian = previousPosition.colPivHouseholderQr().solve(velocityVec);

	Jacobian.transposeInPlace();

	float errorSum = 0.0;
	Vector3f vecError;
	for (int i = 0; i < vecSize; ++i)
	{
		vecError = Jacobian*Vector3f(previousPosition(i,0),previousPosition(i,1),previousPosition(i,2))
				   -Vector3f(velocityVec(i,0),velocityVec(i,1),velocityVec(i,2));
		errorSum += vecError.transpose()*vecError;
	}
	prevTarget[55] = errorSum;

	std::sort(store.begin(), store.end());
	prevTarget[51] = store[7]-store[0];

	a = float(vecSize)/float(8);
	float varience = 0.0;
	for (int i = 0; i < 8; ++i)
	{
		varience+=(store[i]-a)*(store[i]-a);
	}
	varience/=float(8);
	prevTarget[52]=varience;


	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			prevTarget[34+3*i+j] = abs(Jacobian(i,j))<ERROR?0.0:Jacobian(i,j);
		}
	}

	// generate symmetric matrix from Jacobian 
	prevTarget[16] = Jacobian(0,0);
	prevTarget[17] = (Jacobian(0,1)+Jacobian(1,0))/2.0;
	prevTarget[18] = (Jacobian(0,2)+Jacobian(2,0))/2.0;
	prevTarget[19] = (Jacobian(1,0)+Jacobian(0,1))/2.0;
	prevTarget[20] = Jacobian(1,1);
	prevTarget[21] = (Jacobian(1,2)+Jacobian(2,1))/2.0;
	prevTarget[22] = (Jacobian(2,0)+Jacobian(0,2))/2.0;
	prevTarget[23] = (Jacobian(2,1)+Jacobian(1,2))/2.0;
	prevTarget[24] = Jacobian(2,2);

	// generate rotational matrix from Jacobian 
	prevTarget[25] = 0.0;
	prevTarget[26] = (Jacobian(0,1)-Jacobian(1,0))/2.0;
	prevTarget[27] = (Jacobian(0,2)-Jacobian(2,0))/2.0;
	prevTarget[28] = (Jacobian(1,0)-Jacobian(0,1))/2.0;
	prevTarget[29] = 0.0;
	prevTarget[30] = (Jacobian(1,2)-Jacobian(2,1))/2.0;
	prevTarget[31] = (Jacobian(2,0)-Jacobian(0,2))/2.0;
	prevTarget[32] = (Jacobian(2,1)-Jacobian(1,2))/2.0;
	prevTarget[33] = 0.0;

	Matrix3f S, Omega, SST, OOT;
	S << prevTarget[16], prevTarget[17], prevTarget[18], 
		 prevTarget[19], prevTarget[20], prevTarget[21], 
		 prevTarget[22], prevTarget[23], prevTarget[24];
	Omega << prevTarget[25], prevTarget[26], prevTarget[27], 
			 prevTarget[28], prevTarget[29], prevTarget[30], 
			 prevTarget[31], prevTarget[32], prevTarget[33];
	SST = S*S.transpose();
	OOT = Omega*Omega.transpose();
	const float& traceSST = SST(0,0)+SST(1,1)+SST(2,2);
	const float& traceOOT = OOT(0,0)+OOT(1,1)+OOT(2,2);
	const float& q_value = 0.5*(traceOOT*traceOOT-traceSST*traceSST);
	prevTarget[43] = abs(q_value)<ERROR?0.0:q_value;


	// calculate lambda2 value for Jacobian 
	Matrix3f SOmiga = S*S + Omega*Omega;
	EigenSolver<Matrix3f> firstResult(SOmiga);
	vector<float> jacob_;
	jacob_.push_back(firstResult.eigenvalues()[0].real());
	jacob_.push_back(firstResult.eigenvalues()[1].real());
	jacob_.push_back(firstResult.eigenvalues()[2].real());
	std::sort(jacob_.begin(), jacob_.end());
	prevTarget[44] = (abs(jacob_[1])<ERROR?0.0:jacob_[1]);


	// this is computing divergence 
	prevTarget[45] = Jacobian(0,0)+Jacobian(1,1)+Jacobian(2,2);

	// this is computing curl 
	prevTarget[46] = sqrt((Jacobian(2,1)-Jacobian(1,2))*(Jacobian(2,1)-Jacobian(1,2))+(Jacobian(0,2)-Jacobian(2,0))
					*(Jacobian(0,2)-Jacobian(2,0))+(Jacobian(1,0)-Jacobian(0,1))*(Jacobian(1,0)-Jacobian(0,1)));

	const float& determin_ = Jacobian.determinant();

	/* 0 means negative determinant, 1 means positive determinant */
	prevTarget[47] = determin_>=0?1:0;

	float frobinus = 0.0;
	for (int j = 0; j < 3; ++j)
	{
		for (int k = 0; k < 3; ++k)
		{
			frobinus += Jacobian(j,k)*Jacobian(j,k);
		}
	}
	prevTarget[48] = frobinus;

	float frobinus_1 = 0.0, frobinus_2 = 0.0;

	/* frobinus_1 is symmetric norm, frobinus_2 is rotational norm */
	for (int j = 0; j < 9; ++j)
	{
		frobinus_1 += prevTarget[16+j]*prevTarget[16+j];
		frobinus_2 += prevTarget[25+j]*prevTarget[25+j];
	}

	/* 0 means stretching dorminants, and 1 means rotation dorminates */
	prevTarget[49] = frobinus_1 > frobinus_2?0:1;
}


void computeRotationGradient(const int& frame, const int& particle)
{
	const vector<int>& vec = neighbor[particle];

	const int& vecSize = vec.size();

	assert(vecSize!=0);

	MatrixXf previousPosition(vecSize, 3);

	VectorXf rotationVariation(vecSize);

	/* current frame information */
	float* prevTarget = data_information[frame][particle];

	float *prev;

	int index;

	for (int i = 0; i < vecSize; i++)
	{
		index = vec[i];
		prev = data_information[frame][index];
		for (int j = 0; j < 3; ++j)
		{
			previousPosition(i,j) = prev[j] - prevTarget[j];
		}
		rotationVariation(i) = prev[15]-prevTarget[15];
	}

/**********************Compute gradient for rotation **************************************************/
	Vector3f rotationGradient = previousPosition.colPivHouseholderQr().solve(rotationVariation);
	for (int i = 0; i < 3; ++i)
	{
		prevTarget[11+i] = rotationGradient(i);
	}

}