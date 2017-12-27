#pragma once
#ifndef PARTICLE_HPP
#define PARTICLE_HPP

//Standard
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <vector>
#include <map>
#include <algorithm>
#include <glm/glm.hpp>

// Include GLEW
#include <GL/glew.h>
#include <unordered_map>
#include <math.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtx/perpendicular.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

//Custom
#include <shader.hpp>
#include <texture.hpp>
#include <controls.hpp>
#include <objloader.hpp>
#include <vboindexer.hpp>
#include <glerror.hpp>

#include <opencl_utils.hpp>


#define PI 3.1415f

typedef struct ParticleStruct {

	cl_float3 current_position;
	cl_float3 predicted_position;
	cl_float3 velocity;
	cl_float3 delta_p;
	cl_float mass;
	cl_float lambda;
	cl_float rho;
	cl_float C;
	cl_float phase;
	cl_float teardrop;
	cl_float isRigidBody;
	cl_float pencil;
	cl_float isCollidingWithRigidBody;
	cl_int hash;
};

class Particle
{
public:

	Particle();

	Particle(
		glm::vec3 startPosition,
		glm::vec3 predicted_position,
		glm::vec3 velocity,
		glm::vec3 delta_p,
		float mass,
		float lambda,
		float rho,
		float C,
		float phase,
		bool teardrop,
		bool isRigidBody,
		bool pencil,
		bool isCollidingWithRigidBody
		);
	
	//Vec3
	glm::vec3 current_position;			//Posição Atual
	glm::vec3 predicted_position;		//Posição Prevista para passo
	glm::vec3 velocity;
	glm::vec3 delta_p;
	
	//Floats
	float mass;
	float lambda;
	float rho;		//Raw density?
	float C;		//Density constraint?
	float phase;

	//?
	float hash;		//Spatial hash position
	float varx;
	float vary;
	
	//Constraints
	bool teardrop;
	bool isRigidBody;	//Constant for rigid body
	bool pencil;		//?? Something with the walls???
	bool isCollidingWithRigidBody;	//When colliding with rigid body, treats as rigid body

	//Vectors
	std::vector<unsigned int> allNeighbours;			//All neighbours
	std::vector<unsigned int> rigidBodyNeighbours;		//Neighbours that are rigid body
	std::vector<unsigned int> notRigidBodyNeighbours;	//Neighbours that are NOT rigid body

};

//Particle related functions

/* Compares two lists of ParticleStruct */
bool compareParticleStructLists(std::vector<ParticleStruct> l1, std::vector<ParticleStruct> l2);

/* Sequential QuickSort method to order particle vector based on their spatial has value */
//TODO: VERIFY NULLPOINTER ON VALUE AND PARTICLES; VERIFY MATCHING SIZE;
void quickSort(cl_int* value, ParticleStruct* particles, int start, int end);

#endif