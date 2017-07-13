#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

float ****loadData = NULL;
float *stepSize = NULL;
int xSize, ySize, zSize;
float xOrigin, yOrigin, zOrigin;

void loadCylinderFile(void);
void gen_tornado( int xs, int ys, int zs, int time );
const char* FindAndJump(const char* buffer, const char* SearchString);
void loadBernardFile(const int& resolutionX, const int& resolutionY, const int& resolutionZ);
void getLorenzSpeed( float x, float y, float z, float *vxp, float *vyp, float *vzp );
void getLorenzFile();


void printVTK(const int& option);
void printTXT(const int& option);
void getNorm(float* vec, float& velocity);
void getJacobianFiniteDifference(const int& i, const int& j, const int& k);
void getJacobianLeastingSquareFitting(const int& i, const int& j, const int& k);
void performCentral();
void getDeterminant(float* vec, float& velocity);
void initializeMemory();
void deleteMemory();


int main(int argc, char* argv[])
{
    cout << "Error for argument input type! 1)Cylinder 2)Tornado 3)Bernard 4)Lorenz" << endl;
    int option;
    if(argc == 2)
    {
        option = atoi(argv[1]);
    }
    else if(argc == 1)
        option = 1;

    stepSize = new float[3];
    switch(option)
    {
    case 1:
        xSize = 192, ySize = 64, zSize = 48;
        stepSize[0] = 32.0/(float)xSize, stepSize[1] = 8.0/(float)ySize, stepSize[2] = 6.0/(float)zSize;
        xOrigin = 0.0, yOrigin = 0.0, zOrigin = 0.0;
        initializeMemory();
        loadCylinderFile();
        break;

    default:
    case 2:
        xSize = 128, ySize = 128, zSize = 128;
        stepSize[0] = 1.0/float(xSize-1), stepSize[1] = 1.0/float(ySize-1), stepSize[2] = 1.0/float(zSize-1);
        xOrigin = 0.0, yOrigin = 0.0, zOrigin = 0.0;
        initializeMemory();
        gen_tornado(xSize, ySize, zSize, 1000);
        break;

    case 3:
        xSize = 128, ySize = 32, zSize = 64;
        stepSize[0] = 4.0/(float)(xSize-1), stepSize[1] = 1.0/(float)(ySize-1), stepSize[2] = 2.0/(float)(zSize-1);
        xOrigin = 0.0, yOrigin = 0.0, zOrigin = 0.0;
        initializeMemory();
        loadBernardFile(xSize, ySize, zSize);
        break;

    case 4:
        xSize = 64, ySize = 64, zSize = 64;
        stepSize[0] = 60.0/(float)xSize, stepSize[1] = 60.0/(float)ySize, stepSize[2] = 60.0/(float)zSize;
        xOrigin = -30.0, yOrigin = -30.0, zOrigin = -10.0;
        initializeMemory();
        getLorenzFile();
        break;
    }

    performCentral();
    printVTK(option);
    printTXT(option);
    deleteMemory();
    return 0;
}


void initializeMemory()
{
    loadData = new float***[xSize];
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < xSize; ++i)
	{
        loadData[i] = new float**[ySize];
		for (int j = 0; j < ySize; ++j)
		{
            loadData[i][j] = new float*[zSize];
			for (int k = 0; k < zSize; ++k)
			{
				loadData[i][j][k] = new float[36];
			}
		}
	}
	cout << "Memory initialization finished!" << endl;
}


void deleteMemory()
{
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < xSize; ++i)
	{
		for (int j = 0; j < ySize; ++j)
		{
			for (int k = 0; k < zSize; ++k)
			{
				delete[] loadData[i][j][k];
			}
            delete[] loadData[i][j];
		}
        delete[] loadData[i];
	}

    delete[] stepSize;
	cout << "Memory deletion finished!" << endl;
}


void loadCylinderFile(void)
{
    const char* FileName = "../flow_t4048.am";

    FILE* fp = fopen(FileName, "rb");
    if (!fp)
    {
        printf("Could not find %s\n", FileName);
        return ;
    }

    //printf("Reading %s\n", FileName);

    //We read the first 2k bytes into memory to parse the header.
    //The fixed buffer size looks a bit like a hack, and it is one, but it gets the job done.
    char buffer[2048];
    fread(buffer, sizeof(char), 2047, fp);
    buffer[2047] = '\0'; //The following string routines prefer null-terminated strings

    if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"))
    {
        printf("Not a proper AmiraMesh file.\n");
        fclose(fp);
        return ;
    }

    //Find the Lattice definition, i.e., the dimensions of the uniform grid
    int xDim(0), yDim(0), zDim(0);
    sscanf(FindAndJump(buffer, "define Lattice"), "%d %d %d", &xDim, &yDim, &zDim);
    //printf("\tGrid Dimensions: %d %d %d\n", xDim, yDim, zDim);

    //Find the BoundingBox
    float xmin(1.0f), ymin(1.0f), zmin(1.0f);
    float xmax(-1.0f), ymax(-1.0f), zmax(-1.0f);
    sscanf(FindAndJump(buffer, "BoundingBox"), "%g %g %g %g %g %g", &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);

    //Is it a uniform grid? We need this only for the sanity check below.
    const bool bIsUniform = (strstr(buffer, "CoordType \"uniform\"") != NULL);
    //printf("\tGridType: %s\n", bIsUniform ? "uniform" : "UNKNOWN");

    //Type of the field: scalar, vector
    int NumComponents(0);
    if (strstr(buffer, "Lattice { float Data }"))
    {
        //Scalar field
        NumComponents = 1;
    }
    else
    {
        //A field with more than one component, i.e., a vector field
        sscanf(FindAndJump(buffer, "Lattice { float["), "%d", &NumComponents);
    }
    //printf("\tNumber of Components: %d\n", NumComponents);

    //Sanity check
    if (xDim <= 0 || yDim <= 0 || zDim <= 0
        || xmin > xmax || ymin > ymax || zmin > zmax
        || !bIsUniform || NumComponents <= 0)
    {
        printf("Something went wrong\n");
        fclose(fp);
        return ;
    }

    //Find the beginning of the data section
    const long idxStartData = strstr(buffer, "# Data section follows") - buffer;
    if (idxStartData > 0)
    {
        //Set the file pointer to the beginning of "# Data section follows"
        fseek(fp, idxStartData, SEEK_SET);
        //Consume this line, which is "# Data section follows"
        fgets(buffer, 2047, fp);
        //Consume the next line, which is "@1"
        fgets(buffer, 2047, fp);

        //Read the data
        // - how much to read
        const size_t NumToRead = xDim * yDim * zDim * NumComponents;
        // - prepare memory; use malloc() if you're using pure C
        float* pData = new float[NumToRead];
        if (pData)
        {
            // - do it
            const size_t ActRead = fread((void*)pData, sizeof(float), NumToRead, fp);
            // - ok?
            if (NumToRead != ActRead)
            {
                printf("Something went wrong while reading the binary data section.\nPremature end of file?\n");
                delete[] pData;
                fclose(fp);
                return ;
            }

            //Test: Print all data values
            //Note: Data runs x-fastest, i.e., the loop over the x-axis is the innermost
            //printf("\nPrinting all values in the same order in which they are in memory:\n");
            int Idx(0);
            float sum[3]={0.};
            for(int k=0;k<zDim;k++)
            {
                for(int j=0;j<yDim;j++)
                {
                    for(int i=0;i<xDim;i++)
                    {
                        //Note: Random access to the value (of the first component) of the grid point (i,j,k):
                        // pData[((k * yDim + j) * xDim + i) * NumComponents]
                        assert(pData[((k * yDim + j) * xDim + i) * NumComponents] == pData[Idx * NumComponents]);

                        float *middle = loadData[i][j][k];

						middle[3] = i*stepSize[0];
						middle[4] = j*stepSize[1];
						middle[5] = k*stepSize[2];

                        for(int c=0;c<NumComponents;c++)
                        {
                            loadData[i][j][k][c] = pData[Idx * NumComponents + c];
                            //printf("%g ", pData[Idx * NumComponents + c]);
                            sum[c]+=pData[Idx * NumComponents + c];
                        }
                        Idx++;
                    }
                }
            }

            float avg[3]={0.};
            for(int c=0;c<3;c++)
            avg[c] = sum[c] / (zDim*yDim*xDim);

/* from the data format, we see first dimension is X, second dimension is Y, and third dimension is Z */

            for(int k=0;k<zDim;k++)
            {
                for(int j=0;j<yDim;j++)
                {
                    for(int i=0;i<xDim;i++)
                    {                        
                        for(int c=0;c<NumComponents;c++)
                        {
                            loadData[i][j][k][c] -= avg[c];
                        }
                        Idx++;
                    }
                }
            }

            delete[] pData;
        }

    }

    fclose(fp);
    return ;
}

void getLorenzSpeed( float x, float y, float z, float *vxp, float *vyp, float *vzp )
{
    float sigma = 10.0;
    float ro = 28.0;
    float beta = 8.0/3.0;
    *vxp = sigma * (y - x);
    *vyp = (x*(ro - z)) - y;
    *vzp = x*y - beta*z ;    
}

void getLorenzFile()
{
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < xSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < zSize; ++k)
            {
                float* temp = loadData[i][j][k];
                temp[3] = i*stepSize[0], temp[4] = j*stepSize[1], temp[5] = k*stepSize[2];
                getLorenzSpeed(temp[3], temp[4], temp[5], &(temp[0]), &(temp[1]), &(temp[2]));
            }
        }
    }
}

void gen_tornado( int xs, int ys, int zs, int time )
/*
 *  Gen_Tornado creates a vector field of dimension [xs,ys,zs,3] from
 *  a proceedural function. By passing in different time arguements,
 *  a slightly different and rotating field is created.
 *
 *  The magnitude of the vector field is highest at some funnel shape
 *  and values range from 0.0 to around 0.4 (I think).
 *
 *  I just wrote these comments, 8 years after I wrote the function.
 *  
 * Developed by Roger A. Crawfis, The Ohio State University
 *
 */
{
  float x, y, z;
  int ix, iy, iz;
  float r, xc, yc, scale, temp, z0;
  float r2 = 8;
  float SMALL = 0.00000000001;
  float xdelta = 1.0 / (xs-1.0);
  float ydelta = 1.0 / (ys-1.0);
  float zdelta = 1.0 / (zs-1.0);

  float *middle;
  int counter = 0;
  for( iz = 0; iz < zs; iz++ )
  {
     z = iz * zdelta;                        // map z to 0->1
     xc = 0.5 + 0.1*sin(0.04*time+10.0*z);   // For each z-slice, determine the spiral circle.
     yc = 0.5 + 0.1*cos(0.03*time+3.0*z);    //    (xc,yc) determine the center of the circle.
     r = 0.1 + 0.4 * z*z + 0.1 * z * sin(8.0*z); //  The radius also changes at each z-slice.
     r2 = 0.2 + 0.1*z;                           //    r is the center radius, r2 is for damping
     for( iy = 0; iy < ys; iy++ )
     {
        y = iy * ydelta;
        for( ix = 0; ix < xs; ix++ )
        {
            middle = loadData[ix][iy][iz];
            middle[3] = ix*stepSize[0];
            middle[4] = iy*stepSize[1];
            middle[5] = iz*stepSize[2];

            x = ix * xdelta;
            temp = sqrt( (y-yc)*(y-yc) + (x-xc)*(x-xc) );
            scale = fabs( r - temp );
/*
 *  I do not like this next line. It produces a discontinuity 
 *  in the magnitude. Fix it later.
 *
 */
           if ( scale > r2 )
              scale = 0.8 - scale;
           else
              scale = 1.0;
            z0 = 0.1 * (0.1 - temp*z );
           if ( z0 < 0.0 )  z0 = 0.0;
           temp = sqrt( temp*temp + z0*z0 );
            scale = (r + r2 - temp) * scale / (temp + SMALL);
            scale = scale / (1+z);
//               *tornado++ = m_x[ix][iy][iz] = scale * (y-yc) + 0.1*(x-xc);
//               *tornado++ = m_y[ix][iy][iz] = scale * -(x-xc) + 0.1*(y-yc);
//               *tornado++ = m_z[ix][iy][iz] = scale * z0;
           loadData[ix][iy][iz][0] = scale * (y-yc) + 0.1*(x-xc);
           loadData[ix][iy][iz][1] = scale * -(x-xc) + 0.1*(y-yc);;
           loadData[ix][iy][iz][2] = scale * z0;;
        }
     }
  }
}


void loadBernardFile(const int& resolutionX, const int& resolutionY, const int& resolutionZ)
{
    int counter = 0;
    int block[3];
    FILE* pFile2 = fopen ( "../bernard.raw" , "rb" );
    if (pFile2==NULL) {fputs ("File error",stderr); exit (1);}
    fseek( pFile2, 0L, SEEK_SET );

    //Read the data
    // - how much to read
    const size_t NumToRead = resolutionX * resolutionY * resolutionZ * 3;
    // - prepare memory; use malloc() if you're using pure C
    unsigned char* pData = new unsigned char[NumToRead];
    if (pData)
    {
        // - do it
        const size_t ActRead = fread((void*)pData, sizeof(unsigned char), NumToRead, pFile2);
        // - ok?
        if (NumToRead != ActRead)
        {
            printf("Something went wrong while reading the binary data section.\nPremature end of file?\n");
            delete[] pData;
            fclose(pFile2);
            return ;
        }

        //Test: Print all data values
        //Note: Data runs x-fastest, i.e., the loop over the x-axis is the innermost
        //printf("\nPrinting all values in the same order in which they are in memory:\n");
        int Idx(0);
        float tmp[3];
        float *middle;
        for(int k=0;k<resolutionZ;k++)
        {
            for(int j=0;j<resolutionY;j++)
            {
                for(int i=0;i<resolutionX;i++)
                {
                    //Note: Random access to the value (of the first component) of the grid point (i,j,k):
                    // pData[((k * yDim + j) * xDim + i) * NumComponents]
                    assert(pData[((k * resolutionY + j) * resolutionX + i) * 3] == pData[Idx * 3]);

                    middle = loadData[i][j][k];
                    middle[3] = i*stepSize[0];
                    middle[4] = j*stepSize[1];
                    middle[5] = k*stepSize[2];

                    for(int c=0;c<3;c++)
                    {
                        tmp[c] = (float)pData[Idx * 3 + c]/255. - 0.5;                        
                    }
                    float dist = sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]);
                    for(int c=0;c<3;c++)
                    {
                        loadData[i][j][k][c] = tmp[c]/dist;
                        //printf("%g ", pData[Idx * NumComponents + c]);
                    }
                    //printf("\n");
                    Idx++;
                }
            }
        }

        delete[] pData;
    }
    fclose(pFile2);

}

const char* FindAndJump(const char* buffer, const char* SearchString)
{
    const char* FoundLoc = strstr(buffer, SearchString);
    if (FoundLoc) 
    	return FoundLoc + strlen(SearchString);
    return buffer;
}

void printTXT(const int& option)
{
    std::ofstream fout;
    stringstream ss;
    string temp;
    switch(option)
    {
    case 1:
        temp = "Cylinder";
        break;

    case 2:
        temp = "Tornado";
        break;

    case 3:
        temp = "Bernard";
        break;

    case 4:
        temp = "Lorenz";
        break;
    }
    ss << temp << ".txt";
    fout.open( ss.str().c_str(), ios::out);
    if(!fout)
    {
        cout << "Error creating txt file!" << endl;
        exit(-1);
    }

    float *mem;
    for (int i = 0; i < xSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < zSize; ++k)
            {
                mem = loadData[i][j][k];
                for (int ii = 0; ii < 36; ++ii)
                {
                    fout << mem[ii] << " ";
                }
                fout << endl;
            }
        }
    }

    fout.close();
}


void printVTK(const int& option)
{
    std::ofstream fout;
    stringstream ss;
    string temp;
    switch(option)
    {
    case 1:
        temp = "Cylinder";
        break;

    case 2:
        temp = "Tornado";
        break;

    case 3:
        temp = "Bernard";
        break;

    case 4:
        temp = "Lorenz";
        break;
    }
    ss << temp << "_analysis.vtk";
    fout.open( ss.str().c_str(), ios::out);

    fout << "# vtk DataFile Version 3.0" << endl;
    fout << "Volume example" << endl;
    fout << "ASCII" << endl;
    fout << "DATASET STRUCTURED_POINTS" << endl;
    fout << "DIMENSIONS " << xSize << " " << ySize << " " << zSize << endl;
    fout << "ASPECT_RATIO " << stepSize[0] << " " << stepSize[1] << " " << stepSize[2] << endl;
    //fout << "ORIGIN " << stepSize[0] << " " << stepSize[1] << " " << stepSize[2] << endl; 
    fout << "ORIGIN " << xOrigin << " " << yOrigin << " " << zOrigin << endl; 

    fout << "POINT_DATA " << xSize*ySize*zSize << endl;


    fout << "SCALARS velocity float 1" << endl;
    fout << "LOOKUP_TABLE velo_table" << endl;
    float velocity;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                getNorm(&loadData[k][j][i][0], velocity);
                fout << velocity << endl;
            }
        }
    }

    fout << "VECTORS velocityDirection float" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][0] << " " << loadData[k][j][i][1] << " " << loadData[k][j][i][2] << endl;
            }
        }
    }

    fout << "SCALARS RealDeterminant float 1" << endl;
    fout << "LOOKUP_TABLE real_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                getDeterminant(&loadData[k][j][i][6], velocity);
                fout << velocity << endl;
            }
        }
    }

    fout << "SCALARS fittingJacobian float 1" << endl;
    fout << "LOOKUP_TABLE fitting_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                getDeterminant(&loadData[k][j][i][15], velocity);
                fout << velocity << endl;
            }
        }
    }

    fout << "SCALARS differenceDeterminant float 1" << endl;
    fout << "LOOKUP_TABLE differenceDetermi_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][24] << endl;
            }
        }
    }

    fout << "SCALARS difference_Fnorm float 1" << endl;
    fout << "LOOKUP_TABLE differenceFnorm_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][25] << endl;
            }
        }
    }

    fout << "SCALARS absoluteError float 1" << endl;
    fout << "LOOKUP_TABLE differenceMax_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][26] << endl;
            }
        }
    }

    fout << "SCALARS relativeError float 1" << endl;
    fout << "LOOKUP_TABLE relativeError_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][27] << endl;
            }
        }
    }

    fout << "SCALARS differenceError float 1" << endl;
    fout << "LOOKUP_TABLE differenceError_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][28] << endl;
            }
        }
    }

    fout << "SCALARS fittedError float 1" << endl;
    fout << "LOOKUP_TABLE fittedError_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][29] << endl;
            }
        }
    }

    fout << "SCALARS group int 1" << endl;
    fout << "LOOKUP_TABLE group_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][30] << endl;
            }
        }
    }


    fout << "SCALARS errorError float 1" << endl;
    fout << "LOOKUP_TABLE errorError_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][31] << endl;
            }
        }
    }

    fout << "SCALARS diffQValue float 1" << endl;
    fout << "LOOKUP_TABLE diffQvalue_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][32] << endl;
            }
        }
    }

    fout << "SCALARS diffLambda2 float 1" << endl;
    fout << "LOOKUP_TABLE diffLambda2_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][33] << endl;
            }
        }
    }

    fout << "SCALARS fittingQValue float 1" << endl;
    fout << "LOOKUP_TABLE fittingQValue_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][34] << endl;
            }
        }
    }

    fout << "SCALARS fittingLambda2 float 1" << endl;
    fout << "LOOKUP_TABLE fittingLambda2_table" << endl;
    for (int i = 0; i < zSize; ++i)
    {
        for (int j = 0; j < ySize; ++j)
        {
            for (int k = 0; k < xSize; ++k)
            {
                fout << loadData[k][j][i][35] << endl;
            }
        }
    }

    fout.close();
}



void performCentral()
{
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < xSize; ++i)
	 {
	 	for (int j = 0; j < ySize; ++j)
	 	{
	 		for (int k = 0; k < zSize; ++k)
	 		{
	 			getJacobianFiniteDifference(i,j,k);
	 			getJacobianLeastingSquareFitting(i,j,k);
	 		}
	 	}
	 }
}



void getNorm(float* vec, float& velocity)
{
    velocity = sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
}


void getDeterminant(float* vec, float& velocity)
{
	Matrix3f jacobMatrix;
	jacobMatrix << vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8];
	velocity = jacobMatrix.determinant();
}


void getJacobianFiniteDifference(const int& i, const int& j, const int& k)
{
	float *middle = loadData[i][j][k];

	middle[3] = i*stepSize[0];
	middle[4] = j*stepSize[1];
	middle[5] = k*stepSize[2];

	float *left, *right;

    if(i==0)
    {
    	right = loadData[i+1][j][k];

    	for (int kk = 0; kk < 3; ++kk)
    	{
    		middle[6+3*kk] = (right[kk]-middle[kk])/stepSize[0];
    	}
    }
    else if(i==xSize-1)
    {
    	left = loadData[i-1][j][k];

    	for (int kk = 0; kk < 3; ++kk)
    	{
    		middle[6+3*kk] = (middle[kk]-left[kk])/stepSize[0];
    	}
    }
    else
    {
    	right = loadData[i+1][j][k];
    	left = loadData[i-1][j][k];

    	for (int kk = 0;kk < 3; kk++)
    	{
    		middle[6+3*kk] = (right[kk]-left[kk])/2.0/stepSize[0];
    	}
    }

    if(j==0)
    {
    	right = loadData[i][j+1][k];

    	for (int kk = 0; kk < 3; ++kk)
    	{
    		middle[7+3*kk] = (right[kk]-middle[kk])/stepSize[1];
    	}
    }
    else if(j==ySize-1)
    {
    	left = loadData[i][j-1][k];

    	for (int kk = 0; kk < 3; ++kk)
    	{
    		middle[7+3*kk] = (middle[kk]-left[kk])/stepSize[1];
    	}
    }
    else
    {
    	right = loadData[i][j+1][k];
    	left = loadData[i][j-1][k];

    	for (int kk = 0;kk < 3; kk++)
    	{
    		middle[7+3*kk] = (right[kk]-left[kk])/2.0/stepSize[1];
    	}
    }

    if(k==0)
    {
    	right = loadData[i][j][k+1];

    	for (int kk = 0; kk < 3; ++kk)
    	{
    		middle[8+3*kk] = (right[kk]-middle[kk])/stepSize[2];
    	}
    }
    else if(k==zSize-1)
    {
    	left = loadData[i][j][k-1];

    	for (int kk = 0; kk < 3; ++kk)
    	{
    		middle[8+3*kk] = (middle[kk]-left[kk])/stepSize[2];
    	}
    }
    else
    {
    	right = loadData[i][j][k+1];
    	left = loadData[i][j][k-1];

    	for (int kk = 0;kk < 3; kk++)
    	{
    		middle[8+3*kk] = (right[kk]-left[kk])/2.0/stepSize[2];
    	}
    }

    Matrix3f Jacobian, S, Omiga, SST, OOT, SOmiga;
    Jacobian << middle[6], middle[7], middle[8], middle[9], middle[10], middle[11], middle[12], middle[13], middle[14];
    S = 0.5*(Jacobian+Jacobian.transpose());

    Omiga = 0.5*(Jacobian-Jacobian.transpose());

    SST = S*S.transpose();

    OOT = Omiga*Omiga.transpose();

    const float& traceS = SST(0,0)+SST(1,1)+SST(2,2);

    const float& traceO = OOT(0,0)+OOT(1,1)+OOT(2,2);

    middle[32] = 0.5*(traceO*traceO-traceS*traceS);

    SOmiga = S*S+Omiga*Omiga;
    EigenSolver<Matrix3f> result(SOmiga);
    vector<float> jacob_;
    jacob_.push_back(result.eigenvalues()[0].real());
    jacob_.push_back(result.eigenvalues()[1].real());
    jacob_.push_back(result.eigenvalues()[2].real());
    std::sort(jacob_.begin(), jacob_.end());
    middle[33] = (abs(jacob_[1])<1.0e-8?0.0:jacob_[1]);
}


/* generate the least square fitting matrix for the target point */
void getJacobianLeastingSquareFitting(const int& i, const int& j, const int& k)
{

	vector<float*> neighbor;
	float* middle = loadData[i][j][k];
 	
    if(i==0)
    {	
    	neighbor.push_back(&loadData[i+1][j][k][0]);
    }
    else if(i==xSize-1)
    {
    	neighbor.push_back(&loadData[i-1][j][k][0]);
    }
    else
    {
    	neighbor.push_back(&loadData[i+1][j][k][0]);
    	neighbor.push_back(&loadData[i-1][j][k][0]);
    }

    if(j==0)
    {
    	neighbor.push_back(&loadData[i][j+1][k][0]);
    }
    else if(j==ySize-1)
    {
    	neighbor.push_back(&loadData[i][j-1][k][0]);
    }
    else
    {
    	neighbor.push_back(&loadData[i][j+1][k][0]);
    	neighbor.push_back(&loadData[i][j-1][k][0]);
    }

    if(k==0)
    {
    	neighbor.push_back(&loadData[i][j][k+1][0]);
    }
    else if(k==zSize-1)
    {
    	neighbor.push_back(&loadData[i][j][k-1][0]);
    }
    else
    {
    	neighbor.push_back(&loadData[i][j][k+1][0]);
    	neighbor.push_back(&loadData[i][j][k-1][0]);
    }

    MatrixXf fucking(neighbor.size(), 3);
    VectorXf velocity[] = {VectorXf(neighbor.size()), VectorXf(neighbor.size()), VectorXf(neighbor.size())};

    for (int kk=0; kk<neighbor.size();kk++)
    {
    	for (int ii = 0; ii < 3; ++ii)
    	{
    		fucking(kk,ii) = neighbor[kk][ii+3]-middle[ii+3]; 
    		velocity[ii](kk) = neighbor[kk][ii]-middle[ii]; 
    	}
    }

    Vector3f firstRow = fucking.colPivHouseholderQr().solve(velocity[0]);
    Vector3f secondRow = fucking.colPivHouseholderQr().solve(velocity[1]);
    Vector3f thirdRow = fucking.colPivHouseholderQr().solve(velocity[2]);

    Matrix3f realJacobian(3,3), fittingJacobian(3,3);

    for (int ii = 0; ii < 3; ++ii)
    {
    	middle[15+ii] = firstRow(ii);
    	middle[18+ii] = secondRow(ii);
    	middle[21+ii] = thirdRow(ii);
    }

    for (int ii = 0; ii < 3; ++ii)
    {
    	for (int kk = 0; kk < 3; ++kk)
    	{
    		realJacobian(ii,kk) = middle[6+3*ii+kk];
    		fittingJacobian(ii,kk) = middle[15+3*ii+kk];
    	}

    }

    Matrix3f difference = fittingJacobian-realJacobian;
    middle[24] = difference.determinant();

    float summation = 0.0;
    float relative = 0.0;
    float error = 0.0;
    float entry, realEntry;

    for (int ii = 0; ii < 3; ++ii)
    {
    	for (int kk = 0; kk < 3; ++kk)
    	{
    		entry = abs(difference(ii,kk));
    		realEntry = abs(realJacobian(ii,kk));
    		summation += entry*entry; /* calculate F-norm of difference matrix */
    		if(error < entry)
    			error = entry; /* get the largest error */
    		if(realEntry < 1.0e-8)
    			continue;
    		if(relative < entry/realEntry)
    			relative = entry/realEntry; /* get the largest relative error of Jacobian difference matrix */
    	}
    }    
    middle[25] = summation;
    middle[26] = error;
    middle[27] = relative;

    error = 0.0, relative = 0.0;
    MatrixXf OriginV(neighbor.size(),3);
    MatrixXf FittedV(neighbor.size(),3);
    MatrixXf DifferenceV(neighbor.size(),3);

    DifferenceV = fucking*realJacobian;

    FittedV = fucking*fittingJacobian;

    for (int ii = 0; ii < neighbor.size(); ++ii)
    {
    	for (int kk = 0; kk < 3; ++kk)
    	{
    		OriginV(ii,kk) = velocity[kk][ii];
    		relative += (FittedV(ii,kk)-OriginV(ii,kk))*(FittedV(ii,kk)-OriginV(ii,kk));
    		error += (DifferenceV(ii,kk)-OriginV(ii,kk))*(DifferenceV(ii,kk)-OriginV(ii,kk));
    	}
    }

    middle[28] = (error<1.0e-8)?0.0:error; /* difference */
    middle[29] = (relative<1.0e-8)?0.0:relative; /* fitted */
    if(error>=relative)
    	middle[30] = 0;
    else
    	middle[30] = 1;
    middle[31] = relative-error;


    Matrix3f Jacobian, S, Omiga, SST, OOT, SOmiga;
    Jacobian << middle[15], middle[16], middle[17], middle[18], middle[19], middle[20], middle[21], middle[22], middle[23];
    S = 0.5*(Jacobian+Jacobian.transpose());
    Omiga = 0.5*(Jacobian-Jacobian.transpose());
    SST = S*S.transpose();
    OOT = Omiga*Omiga.transpose();
    const float& traceS = SST(0,0)+SST(1,1)+SST(2,2);
    const float& traceO = OOT(0,0)+OOT(1,1)+OOT(2,2);
    middle[34] = 0.5*(traceO*traceO-traceS*traceS);

    SOmiga = S*S+Omiga*Omiga;
    EigenSolver<Matrix3f> result(SOmiga);
    vector<float> jacob_;
    jacob_.push_back(result.eigenvalues()[0].real());
    jacob_.push_back(result.eigenvalues()[1].real());
    jacob_.push_back(result.eigenvalues()[2].real());
    std::sort(jacob_.begin(), jacob_.end());
    middle[35] = (abs(jacob_[1])<1.0e-8?0.0:jacob_[1]);

}