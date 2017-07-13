//#include "ABC_flow.h"
#include "FTLE_Calculator.h"
#include<ctime>

const int FRAME = 100;
const int PARTICLE = 1030301;

const int& T = 10;
float *data_information[PARTICLE];
float readme[4][2] = 
				{	
					FLT_MAX, FLT_MIN, // min radius, max radius
					FLT_MAX, FLT_MIN, // min mlsf ftle, max mlsf ftle
					FLT_MAX, FLT_MIN, // min real ftle, max real ftle
					FLT_MAX, FLT_MIN  // min difference, max difference
				};
float t_time;
int ABCtype;

void initialize_memory_file_reader();
void release_memory_file_reader();
void get_limit();
void generate_readme();

int main(int argc, char **argv)
{
	clock_t a = clock();
	initialize_memory_file_reader();
	const float& time_ = 0.01;

	if(argc < 2)
		ABCtype = 0;
	else
		ABCtype = atoi(argv[1]);
	
	ABC_flow::generate_flow_position(data_information, PARTICLE);
	FTLE_Calculator ftle;
	ftle.get_FTLE_value(data_information, FRAME, PARTICLE, T, time_, ABCtype, readme);
	release_memory_file_reader();
	t_time = (clock() - a)/CLOCKS_PER_SEC;
	std::cout << "The computational time is: " << t_time << " S!" << std::endl;
	generate_readme();
	return 0;
}

void initialize_memory_file_reader()
{
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < PARTICLE; ++i)
	{
		data_information[i] = new float[72];
		/*  the data format is: 
			coordinate: 0, 1, 2
			velocity: 3, 4, 5
			Radius: 6
			LSF_ftle: 7
			real_ftle: 8
			ftle_diff: 9
			LSF_Jacobian: 10, 11, 12, 13, 14, 15, 16, 17, 18
			Real_Jacobian: 19, 20, 21, 22, 23, 24, 25, 26, 27
			Jacobian_diff_determinant: 28
			Jacobian_diff_F_norm: 29
			Jacobian_diff_MaxElement: 30
			realJacobianError: 31
			Fewer_Jacobian: 32, 33, 34, 35, 36, 37, 38, 39, 40, 
			diff_determinant: 41
			diff_F_norm: 42
			diff_max_element: 43
			fittedJacobianError: 44
			fewerJacobianError: 45
			RK-2_position_after_T: 46, 47, 48
			central_Jacobian: 49, 50, 51, 52, 53, 54, 55, 56, 57
			central_ftle: 58
			fullDiff_to_centralMax: 59
			fullDiff_to_centralFNorm: 60
			fewerDiff_to_centralMax: 61
			fewerDiff_to_centralFNorm: 62
			fewer_to_full_lsfMax: 63
			fewer_to_full_lsfFNorm: 64
			fewer_to_full_lsfMaxRelative: 65

			virtual_to_full_lsfMax: 66
			virtual_to_full_lsfFNorm: 67
			virtual_to_full_lsfDeterminant: 68

			virtual_to_centralMax: 69
			virtual_to_central_FNorm: 70
			virtual_to_central_Determinant: 71
		*/
	}
	std::cout << "Memory allocation completed!" << std::endl;
	std::cout << std::endl;
}


void release_memory_file_reader()
{
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < PARTICLE; i++)
	{
		delete[] data_information[i];
	}

	std::cout << "Memory elimination completed! " << std::endl;
	std::cout << std::endl;
}


void generate_readme()
{
	std::ofstream read_("README");
	if (!read_)
	{
		cout << "Error outputting the readme file!" << endl;
		exit(-1);
	}
	read_ << "# ABC Flow Type: simplified flow " << endl;
	read_ << endl;
	read_ << "# Frame: " << FRAME << ", Particle: " << PARTICLE << ", T = " << T << ", neighbor = " << 15 << endl;
	read_ << endl;
	read_ << "# Radius value: [ " << readme[0][0] << ", " << readme[0][1] << "]" << endl;
	read_ << endl;
	read_ << "# Fitted FTLE value: [ " << readme[1][0] << ", " << readme[1][1] << "]" << endl;
	read_ << endl;
	read_ << "# Real FTLE value: [ " << readme[2][0] << ", " << readme[2][1] << "]" << endl;
	read_ << endl;
	read_ << "# Difference value: [ " << readme[3][0] << ", " << readme[3][1] << "]" << endl;
	read_ << endl;
	read_ << "Comptational Time: " << t_time << " s!" << endl;
	read_.close();
}