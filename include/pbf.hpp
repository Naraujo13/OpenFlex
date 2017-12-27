#pragma once
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

#include "particle.hpp"

using namespace std;

typedef std::unordered_multimap< int, int > Hash;


void InitParticleList();
void teardrop();
void rigidBody();
void cube();
void hose();
float wPoly6(glm::vec3 &r, float &h);
glm::vec3 wSpiky(glm::vec3 &r, float &h);
void DensityEstimator(std::vector<ParticleStruct> &predict_p, std::vector<int> neighborBins, int* binBoundaries, int &i);
float NablaCSquaredSumFunction(ParticleStruct &p, std::vector<ParticleStruct> &predict_p, std::vector<int> neighborBins, int* binBoundaries);
glm::vec3 cohesion(ParticleStruct &p, ParticleStruct &p_neighbor);
glm::vec3 surfaceArea(ParticleStruct &p, std::vector <int> neighborBins, cl_int* binBoundaries);
glm::vec3 curvature(ParticleStruct &p, ParticleStruct &p_neighbor, std::vector <int> neighborBins, cl_int* binBoundaries, cl_int* numBins);
glm::vec3 surfaceTension(ParticleStruct &p, std::vector <int> neighborBins, cl_int* binBoundaries, cl_int* numBins);
void CalculateDp(std::vector<ParticleStruct> &predict_p, cl_int* numBins, cl_int* binBoundaries);
glm::vec3 eta(ParticleStruct &p, float &vorticityMag, std::vector<int> neighborBins, cl_int* binBoundaries);
glm::vec3 VorticityConfinement(ParticleStruct &p, std::vector<int> neighborBins, cl_int* binBoundaries);
glm::vec3 XSPHViscosity(ParticleStruct &p, std::vector<int> neighborBins, cl_int* binBoundaries);
void CollisionDetectionResponse(std::vector< ParticleStruct > &p_list);
float boundaryVolume(ParticleStruct p, std::vector<int> neighborBins, cl_int* binBoundaries);
glm::vec3 adhesion(ParticleStruct &p, std::vector <int> neighborBins, cl_int* h_binBoundaries);
glm::vec3 particleFriction(ParticleStruct &p, std::vector <int> neighborBins, cl_int* h_binBoundaries, int i);
void movewallz(std::vector<ParticleStruct> &p_list);
void movewallx(std::vector<ParticleStruct> &p_list);
void movewally(std::vector<ParticleStruct> &p_list);
std::vector<int> getNeighbourBins(int initialBin, cl_int* numBins, int totalBins);

#endif