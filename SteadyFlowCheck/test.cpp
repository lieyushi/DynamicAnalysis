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

struct Point
{
	float position[2];
	float velocity[2];
};

int main()
{

	Vector2f x0, v0;
	x0(0) = x0(1) = 0.0;
	v0(0) = 5, v0(1) = 6;

	Vector2f x1, v1;
	x1(0) = 1, x1(1) = 1;
	v1(0) = 3.6, v1(1) = 4.1;

	Vector2f x2, v2;
	x2(0) = -1, x2(1) = 1;
	v2(0) = 11.4, v2(1) = 2.6;

	Vector2f x3, v3;
	x3(0) = -1, x3(1) = -1;
	v3(0) = 5.6, v3(1) = 7.8;

	Vector2f x4, v4;
	x4(0) = 1, x4(1) = -1;
	v4(0) = -1.4, v4(1) = 10.5;

	Vector2f x5, v5;
	x5(0) = 1.5, x5(1) = 1.0;
	v5(0) = 2.6, v5(1) = 4.1;

	Vector2f x6, v6;
	x6(0) = 2.0, x6(1) = 1.5;
	v6(0) = 1.5, v6(1) = 3.2;

	Vector2f x7, v7;
	x7(0) = 3.5, x7(1) = 2.1;
	v7(0) = -1.8, v7(1) = 4.0;

	MatrixXf distance(4,2);
	for (int i = 0; i < 2; ++i)
	{
		distance(0,i) = x1(i);
		distance(1,i) = x2(i);
		distance(2,i) = x3(i);
		distance(3,i) = x4(i);
	}

	cout << "==================================================" << endl;
	cout << " Solving regular least square by matrix!" << endl;
	MatrixXf target(2,4);
	target(0,0) = v1(0)-v0(0), target(0,1) = v2(0)-v0(0), target(0,2) = v3(0)-v0(0), target(0,3) = v4(0)-v0(0);
	target(1,0) = v1(1)-v0(1), target(1,1) = v2(1)-v0(1), target(1,2) = v3(1)-v0(1), target(1,3) = v4(1)-v0(1);
	MatrixXf temp = (distance.colPivHouseholderQr().solve(target.transpose()));
	temp.transposeInPlace();
	cout << temp << endl;

	VectorXf v_x(4), v_y(4);
	v_x(0) = v1(0)-v0(0), v_x(1) = v2(0)-v0(0), v_x(2) = v3(0)-v0(0), v_x(3) = v4(0)-v0(0);
	v_y(0) = v1(1)-v0(1), v_y(1) = v2(1)-v0(1), v_y(2) = v3(1)-v0(1), v_y(3) = v4(1)-v0(1);

	VectorXf xSolution_1 = distance.colPivHouseholderQr().solve(v_x);
	VectorXf ySolution_1 = distance.colPivHouseholderQr().solve(v_y);

	cout << endl;
	float error = 0.0;
	for (int i = 0; i < 4; ++i)
	{
		Vector2f err = temp*Vector2f(distance(i,0),distance(i,1)) - Vector2f(target(0,i),target(1,i));
		error += err.transpose()*err;
	}
	cout << "Erros is: " << error << endl;

	cout << "==================================================" << endl;
	cout << " Solving fewer neighbors!" << endl;
	MatrixXf another(2,3);
	another(0,0) = v1(0)-v0(0), another(0,1) = v2(0)-v0(0), another(0,2) = v3(0)-v0(0);
	another(1,0) = v1(1)-v0(1), another(1,1) = v2(1)-v0(1), another(1,2) = v3(1)-v0(1);
	MatrixXf second(3,2);
	for (int i = 0; i < 2; ++i)
	{
		second(0,i) = x1(i);
		second(1,i) = x2(i);
		second(2,i) = x3(i);
	}
	Matrix2f temp1 = (second.colPivHouseholderQr().solve(another.transpose()));
	temp1.transposeInPlace();
	cout << temp1 << endl;
	error = 0.0;
	for (int i = 0; i < 3; ++i)
	{
		Vector2f err = temp1*Vector2f(second(i,0),second(i,1)) - Vector2f(another(0,i),another(1,i));
		error += err.transpose()*err;
	}
	cout << "Error is: " << error << endl;

	cout << "==================================================" << endl;
	cout << " Solving more neighbors!" << endl;
	MatrixXf more(2,5);
	more(0,0) = v1(0)-v0(0), more(0,1) = v3(0)-v0(0), more(0,2) = v5(0)-v0(0), more(0,3) = v6(0)-v0(0), more(0,4) = v7(0)-v0(0);
	more(1,0) = v1(1)-v0(1), more(1,1) = v3(1)-v0(1), more(1,2) = v5(1)-v0(1), more(1,3) = v6(1)-v0(1), more(1,4) = v7(1)-v0(1);
	MatrixXf more_(5,2);
	for (int i = 0; i < 2; ++i)
	{
		more_(0,i) = x1(i);
		more_(1,i) = x3(i);
		more_(2,i) = x5(i);
		more_(3,i) = x6(i);
		more_(4,i) = x7(i);
	}
	Matrix2f temp1_ = (more_.colPivHouseholderQr().solve(more.transpose()));
	temp1_.transposeInPlace();
	cout << temp1_ << endl;
	error = 0.0;
	for (int i = 0; i < 5; ++i)
	{
		Vector2f err = temp1_*Vector2f(more_(i,0),more_(i,1)) - Vector2f(more(0,i),more(1,i));
		error += err.transpose()*err;
	}
	cout << "Error is: " << error << endl;
	cout << endl;

	cout << " Solving weighted least square fitting by matrix!" << endl;
	MatrixXf weight = Eigen::DiagonalMatrix<float,5>();
	weight(0,0) = 1.0/4.0, weight(1,1) = 1.0, weight(2,2) = 1.0/4.0, weight(3,3) = 1.0/4.0, weight(4,4) = 1.0/4.0;

	MatrixXf temp2_ = (more_.transpose()*weight*more_).colPivHouseholderQr().solve(more_.transpose()*weight*more.transpose());
	temp2_.transposeInPlace();
	cout << temp2_ << endl;

	error = 0.0;
	for (int i = 0; i < 5; ++i)
	{
		Vector2f err = temp2_*Vector2f(more_(i,0),more_(i,1)) - Vector2f(more(0,i),more(1,i));
		error += err.transpose()*err;
	}
	cout << "Error is: " << error << endl;

	return 0;
}