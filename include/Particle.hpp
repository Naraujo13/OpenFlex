#pragma once
#ifndef PARTICLE_HPP

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <vector>
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


#define PI 3.1415f

class Particle
{
public:

	Particle(
		glm::vec3 startPosition,
		glm::vec3 pred_position,
		glm::vec3 velocity,
		glm::vec3 delta_p,
		float mass,
		float lambda,
		float rho,
		float C,
		float phase,
		bool teardrop,
		bool wall,
		bool pencil,
		bool hybrid
		);
	
	//Vec3
	glm::vec3 position;		// Posição Inicial
	glm::vec3 pred_position;  // Posição Prevista durante o passo
	glm::vec3 velocity;
	glm::vec3 delta_p;
	
	//Floats
	float mass;
	float lambda;
	float rho;
	float C;
	float phase;

	//?
	float hash;
	float varx;
	float vary;
	
	//Constraints
	bool teardrop;
	bool wall;
	bool pencil;
	bool hybrid;

	//Vectors
	std::vector<unsigned int> neighbors;
	std::vector<unsigned int> wneighbors;
	std::vector<unsigned int> allneighbors;

};


//Particle related functions

void InitializeParticleList();


#define PARTICLE_HPP
#endif