#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <omp.h>
// Include GLEW
#include <GL/glew.h>
#include <unordered_map>
#include <math.h>
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtx/perpendicular.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;
#include <shader.hpp>
#include <texture.hpp>
#include <controls.hpp>
#include <objloader.hpp>
#include <vboindexer.hpp>
#include <glerror.hpp>
using namespace std;


#include "pbf.hpp"

Hash hash_table;

std::vector<Particle> particlesList;
std::vector< Particle > predict_p;
//std::vector<float> g_grid;
float g_xmax = 2;
float g_xmin = 0;
float g_ymax = 2;
float g_ymin = 0;
float g_zmax = 2;
float g_zmin = 0;
float h_sphere = 0.05;
float g_h = 0.12;
float POW_H_9 = (g_h*g_h*g_h*g_h*g_h*g_h*g_h*g_h*g_h); // h^9
float POW_H_6 = (g_h*g_h*g_h*g_h*g_h*g_h); // h^6
										   //float wall = 10;
										   //time_t g_initTime = time(0);
float rotatey = 0;
float rotatex = 0;
float rotatez = 0;
float g_k = 0.0011f;
float g_dq = 0.30f;
float gota = 0;
float viscosityC = 0.0095f;
float vorticityEps = 0.00001f;
float resetParticles = 0;
glm::vec3 gravity = glm::vec3(0.0, -9.8, 0.0);
float boundary = 0.03f;
float masswall = 0.5;

float wallscalex = 1;
float wallscaley = 1;
float wallscalez = 1;
float wallLightx = 1;
float wallLighty = 1;
float wallLightz = 1;
int GRID_RESOLUTION = 20;
float adhesioncoeff = 0.5;
float cohesioncoeff = 0.5;
float move_wallz = 0;
float move_wallx = 0;
float move_wally = 0;
float h_adhesion = 0.1;
float gravity_y = -9.8;
float kinetic = 0.5;
float stattc = 0.5;
float REST = 6378.0f;
float wall_h = 0.05;
float solid = 1;

glm::vec3 positions;
glm::vec3 direction;

//bool render = false;
//int npart = PARTICLE_COUNT_X*PARTICLE_COUNT_Y*PARTICLE_COUNT_Z;

//glm::vec3 fext = glm::vec3(0.0, -9.8, 0.0);


//void initializeParticles () {
//	particles.reserve(NPART);
//	srand((unsigned int) 1);
//
//	for (int i = 0 ; i < NPART ; i++){
//		Particle p;
//		p.position.x = (float)rand()/(float)(RAND_MAX/g_xmax);
//		p.position.y = (float)rand()/(float)(RAND_MAX/g_ymax);
//		p.position.z = (float)rand()/(float)(RAND_MAX/g_zmax);;
//
//		p.velocity = glm::vec3(0.0f);
//		p.mass = 1;
//		p.delta_p = glm::vec3(0.0f);
//		p.rho = 0.0;
//		p.C = 1;
//		p.pred_position = glm::vec3(0.0f);
//		p.lambda = 0.0f;
//		
//		particles.push_back(p);
//	}
//};

void InitParticleList()
{
	particlesList.clear();
	//start positioning particles at some distance from the left and bottom walls
	float x_ini_pos = g_xmax / 2 - boundary;
	float y_ini_pos = g_ymin + boundary;
	float z_ini_pos = g_zmax / 2 - boundary;

	// deltas for particle distribution
	float d_x = 0.056f;
	float d_y = 0.056f;
	float d_z = 0.056f;

	printf("Number of particles in the simulation: %i.\n", PARTICLE_COUNT_X*PARTICLE_COUNT_Y*PARTICLE_COUNT_Z);

	float x_pos = x_ini_pos;
	/*particles.reserve(PARTICLE_COUNT_X*PARTICLE_COUNT_Y*PARTICLE_COUNT_Z);*/

	for (unsigned int x = 0; x < PARTICLE_COUNT_X; x++)
	{
		float y_pos = y_ini_pos;

		for (unsigned int y = 0; y < PARTICLE_COUNT_Y; y++)
		{
			float z_pos = z_ini_pos;

			for (unsigned int z = 0; z < PARTICLE_COUNT_Z; z++)
			{
				Particle p;

				float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

				//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


				p.position.x = x_pos + r;
				p.position.y = y_pos + r;
				p.position.z = z_pos + r;

				p.velocity = glm::vec3(0.0f);
				p.mass = 1;
				p.delta_p = glm::vec3(0.0f);
				p.rho = 0.0;
				p.C = 1;
				p.pred_position = glm::vec3(0.0f);
				p.teardrop = false;
				p.wall = false;
				p.hybrid = false;
				p.pencil = false;
				p.lambda = 0.0f;
				p.phase = 0.0f;


				particlesList.push_back(p);
				z_pos += d_z;
			}
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	predict_p = particlesList;
}

void teardrop()
{
	//start positioning particles at some distance from the left and bottom walls
	float x_ini_pos = 0.5;
	float y_ini_pos = 10;
	float z_ini_pos = 0.5;

	// deltas for particle distribution
	float d_x = 0.05f;
	float d_y = 0.05f;
	float d_z = 0.05f;

	float x_pos = x_ini_pos;

	for (unsigned int x = 0; x < 3; x++)
	{
		float y_pos = y_ini_pos;

		for (unsigned int y = 0; y < 5; y++)
		{
			float z_pos = z_ini_pos;

			for (unsigned int z = 0; z < 2; z++)
			{
				Particle p;

				float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

				//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


				p.position.x = x_pos + r;
				p.position.y = y_pos + r;
				p.position.z = z_pos + r;

				p.velocity = glm::vec3(0.0f);
				p.mass = 1;
				p.delta_p = glm::vec3(0.0f);
				p.rho = 0.0;
				p.C = 0;
				p.pred_position = glm::vec3(0.0f);
				p.teardrop = true;
				p.wall = false;
				p.hybrid = false;
				p.lambda = 0.0f;

				particlesList.push_back(p);
				z_pos += d_z;
			}
			y_pos += d_y;
		}
		x_pos += d_x;
	}
	predict_p = particlesList;
}

void wall()
{
	//start positioning particles at some distance from the left and bottom walls
	float x_ini_pos = 0;
	float y_ini_pos = 0;
	float z_ini_pos = 1;

	// deltas for particle distribution
	float d_x = 0.03f;
	float d_y = 0.03f;
	float d_z = 0.03f;
	float limitx = 60;
	float limity = 60;

	float x_pos = x_ini_pos;

	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;

		for (unsigned int y = 0; y < limity; y++)
		{
			Particle p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.position.x = x_pos;
			p.position.y = y_pos;
			p.position.z = 0.6f;

			p.velocity = glm::vec3(0.0f);
			p.mass = masswall;
			p.delta_p = glm::vec3(0.0f);
			p.rho = 0.0;
			p.C = 0;
			p.pred_position = glm::vec3(0.0f);
			p.wall = true;
			p.teardrop = true;
			p.hybrid = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particlesList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	printf("Number of particles in the simulation: %i.\n", particlesList.size());

	predict_p = particlesList;
}

void wall3()
{
	//start positioning particles at some distance from the left and bottom walls
	float x_ini_pos = 1;
	float y_ini_pos = 0;
	float z_ini_pos = 0;

	// deltas for particle distribution
	float d_x = 0.03f;
	float d_y = 0.03f;
	float d_z = 0.03f;
	float limitx = g_xmax * 30;
	float limity = g_ymax * 30;

	float x_pos = x_ini_pos;

	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;

		for (unsigned int y = 0; y < limity; y++)
		{
			Particle p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.position.x = x_pos;
			p.position.y = 0.6f;
			p.position.z = y_pos;

			p.velocity = glm::vec3(0.0f);
			p.mass = masswall;
			p.delta_p = glm::vec3(0.0f);
			p.rho = 0.0;
			p.C = 0;
			p.pred_position = glm::vec3(0.0f);
			p.wall = true;
			p.teardrop = true;
			p.hybrid = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particlesList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	printf("Number of particles in the simulation: %i.\n", particlesList.size());

	predict_p = particlesList;
}

void wall2()
{
	//start positioning particles at some distance from the left and bottom walls
	float x_ini_pos = 0;
	float y_ini_pos = 0;
	float z_ini_pos = 0;

	// deltas for particle distribution
	float d_x = 0.04f;
	float d_y = 0.04f;
	float d_z = 0.04f;
	float limitx = g_xmax * 6;
	float limity = g_ymax * 6;

	float x_pos = x_ini_pos;

	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;

		for (unsigned int y = 0; y < limity; y++)
		{
			Particle p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.position.x = x_pos;
			p.position.y = y_pos;

			printf("variação %f.\n", x_pos);
			p.velocity = glm::vec3(0.0f);
			p.mass = masswall;
			p.delta_p = glm::vec3(0.0f);
			p.rho = 0.0;
			p.C = 0;
			p.pred_position = glm::vec3(0.0f);
			p.wall = true;
			p.pencil = true;
			p.teardrop = true;
			p.hybrid = false;
			p.lambda = 0.0f;
			p.varx = x_pos;
			p.vary = y_pos;
			p.phase = 1.0f;

			particlesList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	printf("Number of particles in the simulation: %i.\n", particlesList.size());

	predict_p = particlesList;
}

void cube()
{
	wall();
	/*wall2();*/
	/*wall3();*/
}

void hose()
{

	//start positioning particles at some distance from the left and bottom walls
	float x_ini_pos = positions.x;
	float y_ini_pos = positions.y;
	float z_ini_pos = 1;
	// deltas for particle distribution
	float d_x = 0.08f;
	float d_y = 0.08f;
	float d_z = 0.08f;

	printf("Number of particles in the simulation: %i.\n", PARTICLE_COUNT_X*PARTICLE_COUNT_Y*PARTICLE_COUNT_Z);

	float x_pos = x_ini_pos;
	/*particles.reserve(PARTICLE_COUNT_X*PARTICLE_COUNT_Y*PARTICLE_COUNT_Z);*/

	for (unsigned int x = 0; x < 2; x++)
	{
		float y_pos = y_ini_pos;

		for (unsigned int y = 0; y < 2; y++)
		{
			float z_pos = z_ini_pos;

			for (unsigned int z = 0; z < 1; z++)
			{
				Particle p;

				float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

				//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


				p.position.x = x_pos;
				p.position.y = y_pos;
				p.position.z = z_pos;

				p.velocity = glm::vec3(0.0f, 0.0f, -2.0f);
				p.mass = 1;
				p.delta_p = glm::vec3(0.0f);
				p.rho = 0.0;
				p.C = 1;
				p.pred_position = glm::vec3(0.0f);
				p.teardrop = false;
				p.wall = false;
				p.pencil = false;
				p.hybrid = false;
				p.lambda = 0.0f;
				p.phase = 0.0f;


				particlesList.insert(particlesList.begin(), p);
				z_pos += d_z;
			}
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	predict_p = particlesList;
}

//SPH Smothing Kernel
float wPoly6(glm::vec3 &r, float &h) {
	float dot_rr = glm::dot(r, r);
	float h2 = h*h;
	float h4 = h2*h2;
	float h2_dot_rr = h2 - dot_rr;

	if (length(r) <= h)
		return 315.0f / (64.0f * 3.1415f * h4*h4*h) * h2_dot_rr * h2_dot_rr * h2_dot_rr;
	else
		return 0.0f;
}

//SPH Kernel Function
glm::vec3 wSpiky(glm::vec3 &r, float &h) {
	float spiky = 45.0f / (PI * pow(h, 6));
	float rlen = glm::length(r);
	if (rlen >= h)
		return glm::vec3(0.0f);
	float coeff = (h - rlen) * (h - rlen);
	coeff *= spiky;
	coeff /= rlen;
	return r * -coeff;
}

void DensityEstimator(std::vector<Particle> &predict_p, int &i) {

	float rhof = 0.0f;
	glm::vec3 r;

	int neighborsize = predict_p[i].allneighbors.size();

	#pragma omp parallel for
	for (int j = 0; j < neighborsize; j++) {
		glm::vec3 r = predict_p[i].position - predict_p[predict_p[i].allneighbors[j]].position;	//Gets position Delta
		rhof += wPoly6(r, g_h); //Accumulate up smoothing kernel(delta, radius) 
	}

	float rhos = 0.0f;

	neighborsize = predict_p[i].wneighbors.size();
	for (int j = 0; j < neighborsize; j++) {
		glm::vec3 r = predict_p[i].position - predict_p[predict_p[i].wneighbors[j]].position;
		rhos += wPoly6(r, g_h);
	}

	predict_p[i].rho = rhof + (solid * rhos);
}

float NablaCSquaredSumFunction(Particle &p, std::vector<Particle> &predict_p) {
	glm::vec3 r;
	std::vector<glm::vec3> NablaC;
	float res = 0.0f;
	int neighborsize = p.neighbors.size();

	if (neighborsize > 0) {
#pragma omp parallel for
		for (int j = 0; j < neighborsize; j++) {													//for k != i
			r = p.position - predict_p[p.neighbors[j]].position;
			glm::vec3 nablac = -wSpiky(r, g_h) / REST;
			NablaC.push_back(nablac);
		}

		NablaC.push_back(glm::vec3(0.0f));																	//for k = i
		int last = NablaC.size() - 1;
#pragma omp parallel for
		for (int j = 0; j < neighborsize; j++) {
			glm::vec3 r = p.position - predict_p[p.neighbors[j]].position;
			NablaC[last] = NablaC[last] + wSpiky(r, g_h);
		}
		NablaC[last] = NablaC[last] / REST;
#pragma omp parallel for
		for (int k = 0; k < NablaC.size(); k++) {
			float norm_nablac = length(NablaC[k]);
			res += norm_nablac * norm_nablac;
		}
	}

	return res;
}

glm::vec3 cohesion(Particle &p, Particle &p_neighbor) {
	glm::vec3 fext = glm::vec3(0.0f);
	float term = 32 / (PI * pow(g_h, 9));
	glm::vec3 r = p.position - p_neighbor.position;
	float normR = length(r);
	float spline = 0;
	if ((2 * normR > g_h) && (normR <= g_h)) {
		float sqrt = (g_h - normR) * (g_h - normR) * (g_h - normR);
		sqrt = sqrt * (normR*normR*normR);
		spline = term * sqrt;
	}

	else if ((2 * normR <= g_h) && (normR > 0)) {
		float term2 = POW_H_6 / 64;
		float sqrt = (g_h - normR) * (g_h - normR) * (g_h - normR);
		sqrt = (2 * sqrt) * (normR*normR*normR);
		spline = term * (sqrt - term2);
	}

	else
		spline = term * 0;
	glm::vec3 divR = r / normR;
	fext = fext + ((-cohesioncoeff)*((p.mass*p_neighbor.mass)*(spline)*(divR)));

	return fext;

}

glm::vec3 surfaceArea(Particle &p, std::vector< Particle > &p_list) {
	glm::vec3 fext = glm::vec3(0.0f);
	unsigned int num_neighbors = p.allneighbors.size();
#pragma omp parallel for
	for (int k = 0; k < num_neighbors; k++) {
		Particle vizinho = p_list[p.allneighbors[k]];
		/*if (!vizinho.wall) {*/
		glm::vec3 r = p.position - p_list[p.allneighbors[k]].position;
		/*glm::vec3 vizinho = p_list[p.allneighbors[k]].position;*/
		float massavizinho = p_list[p.allneighbors[k]].mass;
		float rhovizinho = p_list[p.allneighbors[k]].rho;
		float spline = p_list[p.allneighbors[k]].mass / p_list[p.allneighbors[k]].rho;
		fext = fext + (spline * wSpiky(r, g_h));
		/*}*/
		/*else fext = fext + 0.0f;*/
	}

	return g_h * fext;

}

glm::vec3 curvature(Particle &p, Particle &p_neighbor, std::vector< Particle > &p_list) {
	glm::vec3 fext = glm::vec3(0.0f);

	glm::vec3 ni = surfaceArea(p, p_list);
	glm::vec3 nj = surfaceArea(p_neighbor, p_list);
	fext = fext + (-cohesioncoeff*p.mass*(ni - nj));
	return fext;

}

glm::vec3 surfaceTension(Particle &p, std::vector< Particle > &p_list) {
	glm::vec3 fext = glm::vec3(0.0f);
	unsigned int num_neighbors = p.allneighbors.size();
	#pragma omp parallel for
	for (int k = 0; k < num_neighbors; k++) {
		float kij = (2 * REST) / (p.rho + p_list[p.allneighbors[k]].rho);
		Particle currentng = p_list[p.allneighbors[k]];
		glm::vec3 cohesionNow = cohesion(p, p_list[p.allneighbors[k]]);
		glm::vec3 curvatureNow = curvature(p, p_list[p.allneighbors[k]], p_list);
		fext = fext + (kij * (cohesionNow + curvatureNow));
	}
	return fext;

}

void CalculateDp(std::vector<Particle> &predict_p) {
	float dqMag = g_dq * g_h;
	float kpoly = 315.0f / (64.0f * PI * pow(g_h, 9));
	float wQH = kpoly * pow((g_h * g_h - dqMag * dqMag), 3);
	unsigned int num_particles = predict_p.size();
	#pragma omp parallel for
	for (int i = 0; i < num_particles; i++) {
		if (!predict_p[i].wall && !predict_p[i].hybrid) {
			int neighborsize = predict_p[i].neighbors.size();
			glm::vec3 res = glm::vec3(0.0f);
			glm::vec3 r;


			for (int j = 0; j < neighborsize; j++) {
				r = predict_p[i].position - predict_p[predict_p[i].neighbors[j]].position;
				/*float corr = wPoly6(r, g_h) / wQH;
				corr *= corr * corr * corr;
				float scorr = -g_k * corr;*/
				float lambdaSum = predict_p[i].lambda + predict_p[predict_p[i].neighbors[j]].lambda;
				res += (lambdaSum /*+ scorr*/)* wSpiky(r, g_h);
			}

			predict_p[i].delta_p = res / REST;
		}
	}
}

glm::vec3 eta(Particle &p, std::vector<Particle> &predict_p, float &vorticityMag) {
	glm::vec3 eta = glm::vec3(0.0f);
	int neighborsize = p.neighbors.size();
	if (neighborsize > 0) {
	#pragma omp parallel for
		for (int j = 0; j < neighborsize; j++) {
			glm::vec3 r = p.position - predict_p[p.neighbors[j]].position;
			eta += wSpiky(r, g_h) * vorticityMag;
		}
	}

	return eta;
}

glm::vec3 VorticityConfinement(Particle &p, std::vector< Particle > &p_list) {
	glm::vec3 omega = glm::vec3(0.0f);
	int num_neighbors = p.neighbors.size();

	#pragma omp parallel for
	for (int k = 0; k < num_neighbors; k++) {
		glm::vec3 v_ij = p_list[p.neighbors[k]].velocity - p.velocity;
		glm::vec3 r = p.position - p_list[p.neighbors[k]].position;
		omega += glm::cross(v_ij, wSpiky(r, g_h));
	}

	float omegaLength = glm::length(omega);
	if (omegaLength == 0.0f) {
		//No direction for eta
		return glm::vec3(0.0f);
	}

	glm::vec3 etaVal = eta(p, p_list, omegaLength);
	if (etaVal.x == 0 && etaVal.y == 0 && etaVal.z == 0) {
		//Particle is isolated or net force is 0
		return glm::vec3(0.0f);
	}

	glm::vec3 n = normalize(etaVal);

	return (glm::cross(n, omega) * vorticityEps);
}

glm::vec3 XSPHViscosity(Particle &p, std::vector< Particle > &p_list)
{
	unsigned int num_neighbors = p.neighbors.size();
	glm::vec3 visc = glm::vec3(0.0f);
	if (num_neighbors > 0) {
	#pragma omp parallel for
		for (int k = 0; k < num_neighbors; k++)
		{
			glm::vec3 v_ij = p_list[p.neighbors[k]].velocity - p.velocity;
			glm::vec3 r = p.position - p_list[p.neighbors[k]].position;
			visc += v_ij * wPoly6(r, g_h);
		}
	}
	return visc * viscosityC;
}

void CollisionDetectionResponse(std::vector< Particle > &p_list)
{
	unsigned int num_particles = p_list.size();

	/*glm::vec2 w_min(wall_min_x, wall_min_y);
	glm::vec2 w_max(wall_max_x, wall_max_y);*/
	#pragma omp parallel for
	for (int i = 0; i < num_particles; i++) {
		if (predict_p[i].position.z < g_zmin + boundary) {
			predict_p[i].position.z = g_zmin + boundary;

			/*glm::vec3 normal = glm::vec3(0,0,1);
			predict_p[i].velocity.z = glm::reflect(predict_p[i].velocity, normal).z * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/
		}
		if (predict_p[i].position.z > g_zmax - boundary) {
			predict_p[i].position.z = g_zmax - boundary;
			/*glm::vec3 normal = glm::vec3(0,0,-1);
			predict_p[i].velocity.z = glm::reflect(predict_p[i].velocity, normal).z * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/
		}
		if (predict_p[i].position.y < g_ymin + boundary) {
			predict_p[i].position.y = g_ymin + boundary;
			/*glm::vec3 normal = glm::vec3(0,1,0);
			predict_p[i].velocity.y = glm::reflect(predict_p[i].velocity, normal).y * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/
		}

		if (predict_p[i].position.x < g_xmin + boundary) {
			predict_p[i].position.x = g_xmin + boundary;
			/*glm::vec3 normal = glm::vec3(1,0,0);
			predict_p[i].velocity.x = glm::reflect(predict_p[i].velocity, normal).x * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/

		}
		if (predict_p[i].position.x > g_xmax - boundary) {
			predict_p[i].position.x = g_xmax - boundary;
			/*glm::vec3 normal = glm::vec3(-1,0,0);
			predict_p[i].velocity.x = glm::reflect(predict_p[i].velocity, normal).x * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/

		}
	}
}

int ComputeHash(int &grid_x, int &grid_y, int &grid_z)
{
	int grid = (grid_x + grid_y * GRID_RESOLUTION) + grid_z * (GRID_RESOLUTION * GRID_RESOLUTION);
	/*printf("--> computehash: %i\n", grid);*/
	return grid;
}

//------------------------------------------------------------------------------
void BuildHashTable(std::vector<Particle> &p_list, Hash &hash_table)
{
	int num_particles = p_list.size();
	int grid_x;
	int grid_y;
	int grid_z;

	float cell_size = (g_xmax - g_xmin) / GRID_RESOLUTION;
	/*printf("Grid: %f\n", cell_size);*/

	hash_table.clear();

	for (int i = 0; i < num_particles; i++)
	{
		grid_x = floor(p_list[i].position[0] / cell_size);
		grid_y = floor(p_list[i].position[1] / cell_size);
		grid_z = floor(p_list[i].position[2] / cell_size);

		p_list[i].hash = ComputeHash(grid_x, grid_y, grid_z);

		hash_table.insert(Hash::value_type(p_list[i].hash, i));
		/*printf("Grid:\n");*/

	}
}

//------------------------------------------------------------------------------
void SetUpNeighborsLists(std::vector<Particle> &p_list, Hash &hash_table)
{
	#pragma omp parallel
	{
		int num_particles = p_list.size();

		float cell_size = (g_xmax - g_xmin) / GRID_RESOLUTION;

		int x_idx;
		int y_idx;
		int z_idx;

		int grid_x_min;
		int grid_y_min;
		int grid_x_max;
		int grid_y_max;
		int grid_z_max;
		int grid_z_min;

		int hash;
		#pragma omp parallel for nowait
		for (int i = 0; i < num_particles; i++) {

			p_list[i].neighbors.clear();

			grid_x_min = floor((p_list[i].position[0] - g_h) / cell_size);
			grid_y_min = floor((p_list[i].position[1] - g_h) / cell_size);
			grid_z_min = floor((p_list[i].position[2] - g_h) / cell_size);

			grid_x_max = floor((p_list[i].position[0] + g_h) / cell_size);
			grid_y_max = floor((p_list[i].position[1] + g_h) / cell_size);
			grid_z_max = floor((p_list[i].position[2] + g_h) / cell_size);
			for (z_idx = grid_z_min; z_idx <= grid_z_max; z_idx++) {
				for (y_idx = grid_y_min; y_idx <= grid_y_max; y_idx++) {
					for (x_idx = grid_x_min; x_idx <= grid_x_max; x_idx++) {
							hash = ComputeHash(x_idx, y_idx, z_idx);
							auto its = hash_table.equal_range(hash);

							for (auto it = its.first; it != its.second; ++it) {
								if (it->second != i) {
									if (length(p_list[i].position - p_list[it->second].position) <= g_h) {
										if (p_list[i].wall && p_list[it->second].wall)
											p_list[i].wneighbors.push_back(it->second);
										if (!p_list[i].wall && p_list[it->second].wall) {
											p_list[i].wneighbors.push_back(it->second);
											if (glm::length(p_list[i].position - p_list[it->second].position) < wall_h)
												p_list[i].hybrid = true;
										}
										if (p_list[i].wall && p_list[it->second].hybrid)
											p_list[i].wneighbors.push_back(it->second);
										if (!p_list[i].wall && !p_list[it->second].wall)
											p_list[i].allneighbors.push_back(it->second);
										p_list[i].neighbors.push_back(it->second);
									}
								}
							}
							/*if (length(p_list[i].position - p_list[it->second].position) <= g_h) {
							if ((p_list[i].phase > 0.0f) && (p_list[it->second].phase > 0.0f))
							p_list[i].wneighbors.push_back(it->second);
							if ((p_list[i].phase < 1.0f) && (p_list[it->second].phase > 0.0f)){
							p_list[i].phase = p_list[it->second].phase / 2;
							p_list[i].wneighbors.push_back(it->second);
							}
							if ((p_list[i].phase < 1.0f) && (p_list[it->second].phase < 1.0f))
							p_list[i].allneighbors.push_back(it->second);
							p_list[i].neighbors.push_back(it->second);
							}*/
					}
				}
			}
		}
	}
}

float boundaryVolume(unsigned int &i, std::vector< Particle > &p_list) {
	float massboundary = 0;
	glm::vec3 r;

	int neighborsize = p_list[i].wneighbors.size();
	#pragma omp parallel for
	for (int j = 0; j < neighborsize; j++) {
		glm::vec3 r = p_list[i].position - p_list[p_list[i].wneighbors[j]].position;
		massboundary += wPoly6(r, h_adhesion);
	}
	massboundary = 1 / massboundary;
	massboundary = REST*massboundary;
	return massboundary;
}

glm::vec3 adhesion(Particle &p, std::vector< Particle > &p_list) {
	glm::vec3 fext = glm::vec3(0.0f);
	unsigned int num_neighbors = p.wneighbors.size();
	float term = 0.007 / pow(h_adhesion, 3.25f);
	#pragma omp parallel for
	for (int k = 0; k < num_neighbors; k++) {
		glm::vec3 r = p.position - p_list[p.wneighbors[k]].position;
		float normR = glm::length(r);
		float spline = 0;
		if ((2 * normR > h_adhesion) && (normR <= h_adhesion)) {
			float sqrt = -(4 * (normR * normR)) / h_adhesion;
			sqrt = sqrt + ((6 * normR) - (2 * h_adhesion));
			sqrt = pow(sqrt, 1.0f / 4.0f);
			spline = term * sqrt;
		}
		else
			spline = term * 0;
		glm::vec3 divR = r / normR;
		float boundaryVol = boundaryVolume(p.wneighbors[k], p_list);
		float coeff = adhesioncoeff;
		if (p_list[p.wneighbors[k]].pencil)
			coeff = adhesioncoeff;
		fext = fext + ((-coeff)*(p.mass)*(boundaryVol)*(spline)*(divR));
	}

	return fext;

}

glm::vec3 particleFriction(Particle &p, std::vector< Particle > &p_list, int i) {
	glm::vec3 deltax = glm::vec3(0.0f);
	unsigned int num_neighbors = p.wneighbors.size();
	#pragma omp parallel for
	for (int k = 0; k < 1; k++) {
		glm::vec3 r = p.position - p_list[p.wneighbors[k]].position;
		float distance = length(r) - g_h;
		glm::vec3 n = r / length(r);
		glm::vec3 xi = particlesList[i].position - p.position;
		glm::vec3 xj = particlesList[p.wneighbors[k]].position - p_list[p.wneighbors[k]].position;
		glm::vec3 perp = glm::perp((xi - xj), n);

		float invmass = 1 / p.mass;
		invmass = invmass / (invmass + invmass);
		if (length(perp) < (stattc * distance)) 
		{
			deltax += invmass * perp;
		}
		else
		{
			float mini = min((kinetic*distance) / length(perp), 1.0f);
			deltax += invmass * (perp*mini);
		}

		predict_p[p.wneighbors[k]].velocity = -invmass * deltax;

	}

	/*printf("FRICTION: %f %f %f.\n", deltax.x, deltax.y, deltax.z);*/

	return deltax;

}

void movewallz(std::vector<Particle> &p_list) {
	int num_particles = p_list.size();

	for (int i = 0; i < num_particles; i++) {
		if (p_list[i].pencil) {
			p_list[i].position.z = p_list[i].position.z + positions.z;
		}
	}
}

void movewallx(std::vector<Particle> &p_list) {
	int num_particles = p_list.size();

	for (int i = 0; i < num_particles; i++) {
		if (p_list[i].pencil) {
			p_list[i].position = (positions + glm::vec3(p_list[i].varx, p_list[i].vary, -1.0f));
		}
	}
}

void movewally(std::vector<Particle> &p_list) {
	int num_particles = p_list.size();

	for (int i = 0; i < num_particles; i++) {
		if (p_list[i].pencil) {
			p_list[i].position.y = p_list[i].position.y + move_wally;
		}
	}
}