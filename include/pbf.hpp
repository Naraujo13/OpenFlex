#ifndef PBF_HPP
#define PBF_HPP

#define PI 3.1415f
#define EPSILON 600.0f
#define ITER 2
//#define REST 6378.0f
#define DT 0.0083f
#define PARTICLE_COUNT_X 10
#define PARTICLE_COUNT_Y 2
#define PARTICLE_COUNT_Z 10
#include <unordered_map>
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

#include "particle.hpp"

using namespace std;

typedef std::unordered_multimap< int, int > Hash;


class ParticleOld
{
public:
	glm::vec3 current_position;		  // Posição Inicial
	glm::vec3 predicted_position;  // Posição Prevista durante o passo
	glm::vec3 velocity;
	glm::vec3 delta_p;
	float mass;
	float lambda;
	float rho;		//Raw density?
	float C;		//Density constraint?
	float hash;		//Hashing value to find neighbours
	bool teardrop;
	bool isRigidBody;		//Is a wall?
	bool pencil;
	bool isCollidingWithRigidBody;	//Near a wall?
	std::vector<unsigned int> allNeighbours;	//All neighbours particles
	std::vector<unsigned int> rigidBodyNeighbours;	//Neigbours particles that are wall
	std::vector<unsigned int> notRigidBodyNeighbours;	//Neighbour particles that are not wall
	float varx;
	float vary;
	float phase;
};

void InitParticleList();
void InitParticleStructList();
void teardrop();
void rigidBody();
void wall3();
void wall2();
void wallStruct();
void cube();
void cubeStruct();
void hose();
float wPoly6(glm::vec3 &r, float &h);
glm::vec3 wSpiky(glm::vec3 &r, float &h);
void DensityEstimator(std::vector<Particle> &predict_p, int &i);
float NablaCSquaredSumFunction(Particle &p, std::vector<Particle> &predict_p);
glm::vec3 cohesion(Particle &p, Particle &p_neighbor);
glm::vec3 surfaceArea(Particle &p, std::vector< Particle > &p_list);
glm::vec3 curvature(Particle &p, Particle &p_neighbor, std::vector< Particle > &p_list);
glm::vec3 surfaceTension(Particle &p, std::vector< Particle > &p_list);
void CalculateDp(std::vector<Particle> &predict_p);
glm::vec3 eta(Particle &p, std::vector<Particle> &predict_p, float &vorticityMag);
glm::vec3 VorticityConfinement(Particle &p, std::vector< Particle > &p_list);
glm::vec3 XSPHViscosity(Particle &p, std::vector< Particle > &p_list);
void CollisionDetectionResponse(std::vector< Particle > &p_list);
int ComputeHash(int &grid_x, int &grid_y, int &grid_z);
void BuildHashTable(std::vector<Particle> &p_list, Hash &hash_table);
void SetUpNeighborsLists(std::vector<Particle> &p_list, Hash &hash_table);
float boundaryVolume(unsigned int &i, std::vector< Particle > &p_list);
glm::vec3 adhesion(Particle &p, std::vector< Particle > &p_list);
glm::vec3 particleFriction(Particle &p, std::vector< Particle > &p_list, int i);
void movewallz(std::vector<Particle> &p_list);
void movewallx(std::vector<Particle> &p_list);
void movewally(std::vector<Particle> &p_list);


void newBuildHashTable(std::vector<Particle> &p_list, SpatialHash &hash_table);
void newSetUpNeighborsLists(std::vector<Particle> &p_list, SpatialHash &hash_table);
#endif