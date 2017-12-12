#pragma once
#ifndef PARTICLE_HPP

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <vector>
#include <map>
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
#include <shader.hpp>
#include <texture.hpp>
#include <controls.hpp>
#include <objloader.hpp>
#include <vboindexer.hpp>
#include <glerror.hpp>
#include<CL\cl.hpp>


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

struct Vec3Comparator {
	size_t operator()(const glm::vec3& v)const {
		return std::hash<float>()(v.x) ^ std::hash<float>()(v.y) ^ std::hash<float>()(v.z);
	}
	
	bool operator()(const glm::vec3& v1, glm::vec3& v2)const {
		return glm::all(glm::lessThan(v1, v2));
	}
};

typedef std::vector<int> particleVector;
typedef std::unordered_map<glm::vec3, particleVector, Vec3Comparator> spatialCell;


class SpatialHash
{

private:
	
	//Variables
	glm::vec3  dim;
	float cellSize;
	spatialCell cells;

	//Clean Hash
	void cleanHash();
	
	//Build Hash
	void buildHash(glm::vec3, float cellSize);

public:

	//Hashing
	glm::vec3 hashFunction(glm::vec3 particlePosition);


	//Constructor
	SpatialHash::SpatialHash();
	SpatialHash::SpatialHash(glm::vec3 dim, float cellSize);

	//Rebuild
	void reconstruct();
	void reconstruct(glm::vec3, float cellSize);

	//Insert
	bool insert(glm::vec3 particlePos, int particleIndex);

	//Get Cell
	std::vector<int> getCell(glm::vec3 index);

	spatialCell getCells() {
		return cells;
	}

};

//Particle related functions

void InitializeParticleList();


#define PARTICLE_HPP
#endif