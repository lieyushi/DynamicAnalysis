#include "Kernel.h"

const float Kernel::W_cubic(const glm::vec3& r, const float& radius)
{
	const float& q = glm::length(r)/radius;
	if(q<1.0e-5 || q >1.0)
		return 0.0;
	else if (q <= 0.5)
	{
		return 2.54647913/pow(radius,3.0)*(6.0*pow(q,3.0)-6.0*pow(q,2.0)+1.0);
	}
	else if(q <= 1.0 && q >0.5)
	{
		return 2.54647913/pow(radius,3.0)*(2.0*pow(1.0-q, 3.0));
	}
}

const float Kernel::W_cubic(const float& r, const float& radius)
{
	const float& q = r/radius;
	if(q<1.0e-5||q>1.0)
		return 0.0;
	else if (q <= 0.5)
	{
		return 2.54647913/pow(radius,3.0)*(6.0*pow(q,3.0)-6.0*pow(q,2.0)+1.0);
	}
	else if(q <= 1.0 && q >0.5)
	{
		return 2.54647913/pow(radius,3.0)*(2.0*pow(1.0-q, 3.0));
	}
}



const glm::vec3 Kernel::W_cubic_gradient(const glm::vec3& r, const float& radius)
{
	const float& q = glm::length(r)/radius;
	float temp;
	if(q < 1.0e-5||q>1.0)
		temp = 0.0;
	else if (q <= 0.5)
	{
		temp = 15.27887477/pow(radius,3.0)*(3.0*pow(q,2.0)-2.0*q)/radius/glm::length(r);
	}
	else if(q <= 1.0 && q > 0.5)
	{
		temp = -15.27887479/pow(radius,3.0)*pow(1-q,2.0)/radius/glm::length(r);
	}
	return glm::vec3(temp*r.x, temp*r.y, temp*r.z);
}


const float Kernel::W_poly6(const glm::vec3& r, const float& radius)
{
	const float& dist = glm::length(r);
	const float& q = dist/radius;
	if(q<1.0e-5||q>1.0)
		return 0.0;
	else if( q <= 1.0)
	{
		return (float)(1.56668149/pow(radius, 9)*pow((pow(radius, 2) - pow(dist, 2)), 3));
	}
}


const float Kernel::W_poly6(const float& r, const float& radius)
{
	const float& q = r/radius;
	if(q < 1.0e-5||q>1.0)
		return 0.0;
	else if( q <= 1.0)
	{
		return (float)(1.56668149/pow(radius, 9)*pow((pow(radius, 2) - pow(r, 2)), 3));
	}
}


const float Kernel::W_spiky(const glm::vec3& r, const float& radius)
{
	const float& dist = glm::length(r);
	const float& q = dist/radius;
	if(q < 1.0e-5||q>1.0)
		return 0.0;
	else if( q <= 1.0)
	{
		return (float)(4.77464837/pow(radius, 6)*pow( radius - dist, 3));
	}
}


const float Kernel::W_spiky(const float& dist, const float& radius)
{
	const float& q = dist/radius;
	if(q < 1.0e-5 || q > 1.0)
		return 0.0;
	else if( q <= 1.0)
	{
		return (float)(4.77464837/pow(radius, 6)*pow( radius - dist, 3));
	}
}


const glm::vec3 Kernel::W_poly6_gradient(const glm::vec3& r, const float& radius)
{
	glm::vec3 gradient;
	const float& dist = glm::length(r);
	const float& q = dist/radius;
	if(q<1.0e-5 || q >1.0)
		return glm::vec3(0.0);
	else if( q <= 1.0)
	{
		float temp = (float)(4.70004447/pow(radius, 9)*pow(pow(radius, 2) - pow(dist, 2), 2))*(-2);
		gradient.x = temp * r.x;
		gradient.y = temp * r.y;
		gradient.z = temp * r.z;
		return gradient;
	}
}

const glm::vec3 Kernel::W_spiky_gradient(const glm::vec3& r, const float& radius)
{
	glm::vec3 gradient;
	const float& dist = glm::length(r);
	const float& q = dist/radius;
	if(q<1.0e-5 || q >1.0)
		return glm::vec3(0.0);
	else if(q <= 1.0)
	{
		float temp = - (float)(14.32394511/pow(radius, 6)*pow(radius - dist, 2));
		gradient.x = temp * r.x/dist;
		gradient.y = temp * r.y/dist;
		gradient.z = temp * r.z/dist;
	}
	return gradient;
}

const float Kernel::W_cubic_laplacian(const glm::vec3& r, const float& radius)
{
	const float& q = glm::length(r)/radius;
	if(q<1.0e-5||q>1.0)
		return 0.0;
	else if(q >0 && q <= 0.5)
		return 15.27887479/pow(radius,5.0)*(6.0*q-2);
	else if(q > 0.5 && q <= 1.0)
		return -15.27887479/pow(radius,5.0)*(2.0-2.0*q);
}


const float Kernel::W_cohesion(const glm::vec3& r, const float& radius)
{
	const float& r_length = glm::length(r);
	const float& q = r_length/radius;
	if(q<1.0e-5||q>1)
		return 0.0;
	else if(q<0.5)
		return 10.1859/pow(radius,9.0)*(2.0*pow(radius-r_length,2.0)*pow(r_length,3.0)-pow(radius,6.0)/64.0);
	else
		return 10.1859*pow(radius-r_length,3.0)*pow(r_length,3.0);
}