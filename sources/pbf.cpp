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

#include <glfw3.h>
#include "pbf.hpp"

Hash hash_table;


std::vector<ParticleStruct> particleStructList;
std::vector<ParticleStruct> predictedStructList;
std::vector<Particle> particlesList;
std::vector< Particle > predict_p;

float g_xmax = 2;
float g_xmin = 0;
float g_ymax = 2;
float g_ymin = 0;
float g_zmax = 2;
float g_zmin = 0;
float particle_size = 0.05;
float g_h = 0.10;
float POW_H_9 = (g_h*g_h*g_h*g_h*g_h*g_h*g_h*g_h*g_h); // h^9
float POW_H_6 = (g_h*g_h*g_h*g_h*g_h*g_h); // h^6
										   //float rigidBody = 10;
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
float adhesioncoeff = 0.0;
float cohesioncoeff = 0.1;
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

//Initialize some water particles - LEGACY
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
	#pragma omp parallel for
	for (unsigned int x = 0; x < PARTICLE_COUNT_X; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < PARTICLE_COUNT_Y; y++)
		{
			float z_pos = z_ini_pos;
			#pragma omp parallel for
			for (unsigned int z = 0; z < PARTICLE_COUNT_Z; z++)
			{
				Particle p;

				float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

				//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


				p.current_position.x = x_pos + r;
				p.current_position.y = y_pos + r;
				p.current_position.z = z_pos + r;

				p.velocity = glm::vec3(0.0f);
				p.mass = 1;
				p.delta_p = glm::vec3(0.0f);
				p.rho = 0.0;
				p.C = 1;
				p.predicted_position = glm::vec3(0.0f);
				p.teardrop = false;
				p.isRigidBody = false;
				p.isCollidingWithRigidBody = false;
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

//Initialize water particles - PORT
void InitParticleStructList()
{
	particleStructList.clear();
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
	#pragma omp parallel for
	for (unsigned int x = 0; x < PARTICLE_COUNT_X; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < PARTICLE_COUNT_Y; y++)
		{
			float z_pos = z_ini_pos;
			#pragma omp parallel for
			for (unsigned int z = 0; z < PARTICLE_COUNT_Z; z++)
			{
				ParticleStruct p;

				float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

				//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


				p.current_position.x = x_pos + r;
				p.current_position.y = y_pos + r;
				p.current_position.z = z_pos + r;

				p.velocity.x = 0;
				p.velocity.y = 0;
				p.velocity.z = 0;

				p.mass = 1;

				p.delta_p.x = 0;
				p.delta_p.y = 0;
				p.delta_p.z = 0;

				p.rho = 0.0;
				p.C = 1;

				p.predicted_position.x = 0;
				p.predicted_position.y = 0;
				p.predicted_position.z = 0;
				
				p.teardrop = false;
				p.isRigidBody = false;
				p.isCollidingWithRigidBody = false;
				p.pencil = false;
				p.lambda = 0.0f;
				p.phase = 0.0f;

				particleStructList.push_back(p);

				z_pos += d_z;
			}
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	predictedStructList = particleStructList;
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
	#pragma omp parallel for
	for (unsigned int x = 0; x < 3; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < 5; y++)
		{
			float z_pos = z_ini_pos;
			#pragma omp parallel for
			for (unsigned int z = 0; z < 2; z++)
			{
				Particle p;

				float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

				//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


				p.current_position.x = x_pos + r;
				p.current_position.y = y_pos + r;
				p.current_position.z = z_pos + r;

				p.velocity = glm::vec3(0.0f);
				p.mass = 1;
				p.delta_p = glm::vec3(0.0f);
				p.rho = 0.0;
				p.C = 0;
				p.predicted_position = glm::vec3(0.0f);
				p.teardrop = true;
				p.isRigidBody = false;
				p.isCollidingWithRigidBody = false;
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

//Creates cube rigid body - LEGACY
void cube()
{
	//start positioning particles at some distance from the left and bottom walls
	float x_ini_pos = 0.8;
	float y_ini_pos = 0;
	float z_ini_pos = 0.2;

	// deltas for particle distribution
	float d_x = 0.03f;
	float d_y = 0.03f;
	float d_z = 0.03f;
	float limitx = 20;
	float limity = 20;

	float x_pos = x_ini_pos;
	float z_pos = z_ini_pos;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			Particle p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity = glm::vec3(0.0f);
			p.mass = masswall;
			p.delta_p = glm::vec3(0.0f);
			p.rho = 0.0;
			p.C = 0;
			p.predicted_position = glm::vec3(0.0f);
			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particlesList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	//2
	x_pos = x_ini_pos;
	z_pos = z_pos+-0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
	#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			Particle p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity = glm::vec3(0.0f);
			p.mass = masswall;
			p.delta_p = glm::vec3(0.0f);
			p.rho = 0.0;
			p.C = 0;
			p.predicted_position = glm::vec3(0.0f);
			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particlesList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	//3
	x_pos = x_ini_pos;
	z_pos = z_pos+0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
	#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			Particle p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity = glm::vec3(0.0f);
			p.mass = masswall;
			p.delta_p = glm::vec3(0.0f);
			p.rho = 0.0;
			p.C = 0;
			p.predicted_position = glm::vec3(0.0f);
			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particlesList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	//4
	x_pos = x_ini_pos;
	z_pos = z_pos+0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			Particle p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity = glm::vec3(0.0f);
			p.mass = masswall;
			p.delta_p = glm::vec3(0.0f);
			p.rho = 0.0;
			p.C = 0;
			p.predicted_position = glm::vec3(0.0f);
			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particlesList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}
	
	//5
	x_pos = x_ini_pos;
	z_pos = z_pos+0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			Particle p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity = glm::vec3(0.0f);
			p.mass = masswall;
			p.delta_p = glm::vec3(0.0f);
			p.rho = 0.0;
			p.C = 0;
			p.predicted_position = glm::vec3(0.0f);
			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particlesList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	//6
	x_pos = x_ini_pos;
	z_pos = z_pos + 0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			Particle p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity = glm::vec3(0.0f);
			p.mass = masswall;
			p.delta_p = glm::vec3(0.0f);
			p.rho = 0.0;
			p.C = 0;
			p.predicted_position = glm::vec3(0.0f);
			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
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

//Creates rigid body cube - PORT
void cubeStruct()
{
	//start positioning particles at some distance from the left and bottom walls
	float x_ini_pos = 0.8;
	float y_ini_pos = 0;
	float z_ini_pos = 0.2;

	// deltas for particle distribution
	float d_x = 0.03f;
	float d_y = 0.03f;
	float d_z = 0.03f;
	float limitx = 20;
	float limity = 20;

	float x_pos = x_ini_pos;
	float z_pos = z_ini_pos;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			ParticleStruct p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;
			
			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity.x = 0;
			p.velocity.y = 0;
			p.velocity.z = 0;

			p.mass = masswall;

			p.delta_p.x = 0;
			p.delta_p.y = 0;
			p.delta_p.z = 0;

			p.rho = 0.0;
			p.C = 0;

			p.predicted_position.x = 0;
			p.predicted_position.y = 0;
			p.predicted_position.z = 0;

			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particleStructList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	//2
	x_pos = x_ini_pos;
	z_pos = z_pos + -0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			ParticleStruct p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity.x = 0;
			p.velocity.y = 0;
			p.velocity.z = 0;

			p.mass = masswall;

			p.delta_p.x = 0;
			p.delta_p.y = 0;
			p.delta_p.z = 0;

			p.rho = 0.0;
			p.C = 0;

			p.predicted_position.x = 0;
			p.predicted_position.y = 0;
			p.predicted_position.z = 0;

			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particleStructList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	//3
	x_pos = x_ini_pos;
	z_pos = z_pos + 0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			ParticleStruct p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity.x = 0;
			p.velocity.y = 0;
			p.velocity.z = 0;

			p.mass = masswall;

			p.delta_p.x = 0;
			p.delta_p.y = 0;
			p.delta_p.z = 0;

			p.rho = 0.0;
			p.C = 0;

			p.predicted_position.x = 0;
			p.predicted_position.y = 0;
			p.predicted_position.z = 0;

			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particleStructList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	//4
	x_pos = x_ini_pos;
	z_pos = z_pos + 0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			ParticleStruct p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity.x = 0;
			p.velocity.y = 0;
			p.velocity.z = 0;

			p.mass = masswall;

			p.delta_p.x = 0;
			p.delta_p.y = 0;
			p.delta_p.z = 0;

			p.rho = 0.0;
			p.C = 0;

			p.predicted_position.x = 0;
			p.predicted_position.y = 0;
			p.predicted_position.z = 0;

			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particleStructList.push_back(p);
			y_pos += d_y;

		}
		x_pos += d_x;
	}

	//5
	x_pos = x_ini_pos;
	z_pos = z_pos + 0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			ParticleStruct p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;


			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity.x = 0;
			p.velocity.y = 0;
			p.velocity.z = 0;

			p.mass = masswall;

			p.delta_p.x = 0;
			p.delta_p.y = 0;
			p.delta_p.z = 0;

			p.rho = 0.0;
			p.C = 0;

			p.predicted_position.x = 0;
			p.predicted_position.y = 0;
			p.predicted_position.z = 0;

			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particleStructList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	//6
	x_pos = x_ini_pos;
	z_pos = z_pos + 0.1;
	#pragma omp parallel for
	for (unsigned int x = 0; x < limitx; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < limity; y++)
		{
			ParticleStruct p;

			float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

			//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


			p.current_position.x = x_pos;
			p.current_position.y = y_pos;
			p.current_position.z = z_pos;

			p.velocity.x = 0;
			p.velocity.y = 0;
			p.velocity.z = 0;

			p.mass = masswall;

			p.delta_p.x = 0;
			p.delta_p.y = 0;
			p.delta_p.z = 0;

			p.rho = 0.0;
			p.C = 0;

			p.predicted_position.x = 0;
			p.predicted_position.y = 0;
			p.predicted_position.z = 0;

			p.isRigidBody = true;
			p.teardrop = true;
			p.isCollidingWithRigidBody = false;
			p.lambda = 0.0f;
			p.phase = 1.0f;

			particleStructList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	printf("Number of particles in the simulation: %i.\n", particleStructList.size());

	predictedStructList = particleStructList;
}

//Hose method, spawn new particles 
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
	
	#pragma omp parallel for
	for (unsigned int x = 0; x < 2; x++)
	{
		float y_pos = y_ini_pos;
		#pragma omp parallel for
		for (unsigned int y = 0; y < 2; y++)
		{
			float z_pos = z_ini_pos;
			#pragma omp parallel for
			for (unsigned int z = 0; z < 1; z++)
			{
				Particle p;

				float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

				p.current_position.x = x_pos;
				p.current_position.y = y_pos;
				p.current_position.z = z_pos;

				p.velocity = glm::vec3(0.0f, 0.0f, -2.0f);
				p.mass = 1;
				p.delta_p = glm::vec3(0.0f);
				p.rho = 0.0;
				p.C = 1;
				p.predicted_position = glm::vec3(0.0f);
				p.teardrop = false;
				p.isRigidBody = false;
				p.pencil = false;
				p.isCollidingWithRigidBody = false;
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

	//Dot product of distance btwn particles
	float dot_rr = glm::dot(r, r);
	
	//Smooth ratio^2
	float h2 = h*h;
	
	//Smooth ratio^4
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

//Estimate rho for particle i 
void DensityEstimator(std::vector<Particle> &predict_p, int &i) {

	float rhof = 0.0f;
	glm::vec3 r;

	int neighborsize = predict_p[i].notRigidBodyNeighbours.size();

	//For each neighbour that is not a rigidBody
	#pragma omp parallel for
	for (int j = 0; j < neighborsize; j++) {
		//Gets distance to neighbour
		glm::vec3 r = predict_p[i].current_position - predict_p[predict_p[i].notRigidBodyNeighbours[j]].current_position;	
		//Accumulate smoothing kernel(delta, radius) 
		rhof += wPoly6(r, g_h);
	}

	float rhos = 0.0f;

	neighborsize = predict_p[i].rigidBodyNeighbours.size();
	
	//For each neighbour that is a rigidBody
	#pragma omp parallel for
	for (int j = 0; j < neighborsize; j++) {
		//Gets distance to neighbour
		glm::vec3 r = predict_p[i].current_position - predict_p[predict_p[i].rigidBodyNeighbours[j]].current_position;
		//Accumulate smoothing kernel(delta, radius) 
		rhos += wPoly6(r, g_h);	
	}
	
	//Set rho for analyzed particle
	predict_p[i].rho = rhof + (solid * rhos);
}

//Nabla Squared Sum - needs revision LEGACY
float NablaCSquaredSumFunction(Particle &p, std::vector<Particle> &predict_p) {
	glm::vec3 r;
	std::vector<glm::vec3> NablaC;
	float res = 0.0f;
	int neighborsize = p.allNeighbours.size();

	//If particle has neighbours
	if (neighborsize > 0) {

		//For each neighbour (all neighbours)
		#pragma omp parallel for
		for (int j = 0; j < neighborsize; j++) {													//for k != i

			//Gets distance to neighbour
			r = p.current_position - predict_p[p.allNeighbours[j]].current_position;

			//Applies kernel and divides by rest?
			glm::vec3 nablac = -wSpiky(r, g_h) / REST;
			
			NablaC.push_back(nablac);
		}

		NablaC.push_back(glm::vec3(0.0f));																	//for k = i
		int last = NablaC.size() - 1;
		
		//For each neighbour
		#pragma omp parallel for
		for (int j = 0; j < neighborsize; j++) {

			//Gets distance to neighbour
			glm::vec3 r = p.current_position - predict_p[p.allNeighbours[j]].current_position;

			//Sums distance in the last position of Nabla vector
			NablaC[last] = NablaC[last] + wSpiky(r, g_h);
		}
		
		//Divides last position for REST
		NablaC[last] = NablaC[last] / REST;
		
		//For each element in Nabla vector
		#pragma omp parallel for
		for (int k = 0; k < NablaC.size(); k++) {
			float norm_nablac = length(NablaC[k]);
			res += norm_nablac * norm_nablac;
		}
	}
	return res;
}

//Calculates coesion
glm::vec3 cohesion(Particle &p, Particle &p_neighbor) {
	glm::vec3 fext = glm::vec3(0.0f);
	float term = 32 / (PI * pow(g_h, 9));
	glm::vec3 r = p.current_position - p_neighbor.current_position;
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

//Calculates surface area
glm::vec3 surfaceArea(Particle &p, std::vector< Particle > &p_list) {
	glm::vec3 fext = glm::vec3(0.0f);
	unsigned int num_neighbors = p.notRigidBodyNeighbours.size();
	
	#pragma omp parallel for
	for (int k = 0; k < num_neighbors; k++) {
		Particle vizinho = p_list[p.notRigidBodyNeighbours[k]];
		/*if (!vizinho.rigidBody) {*/
		glm::vec3 r = p.current_position - p_list[p.notRigidBodyNeighbours[k]].current_position;
		/*glm::vec3 vizinho = p_list[p.notRigidBodyNeighbours[k]].position;*/
		float massavizinho = p_list[p.notRigidBodyNeighbours[k]].mass;
		float rhovizinho = p_list[p.notRigidBodyNeighbours[k]].rho;
		float spline = p_list[p.notRigidBodyNeighbours[k]].mass / p_list[p.notRigidBodyNeighbours[k]].rho;
		fext = fext + (spline * wSpiky(r, g_h));
		/*}*/
		/*else fext = fext + 0.0f;*/
	}

	return g_h * fext;

}

//Calculates curvature
glm::vec3 curvature(Particle &p, Particle &p_neighbor, std::vector< Particle > &p_list) {
	glm::vec3 fext = glm::vec3(0.0f);

	glm::vec3 ni = surfaceArea(p, p_list);
	glm::vec3 nj = surfaceArea(p_neighbor, p_list);
	fext = fext + (-cohesioncoeff*p.mass*(ni - nj));
	return fext;

}

//Caclulates surface tension
glm::vec3 surfaceTension(Particle &p, std::vector< Particle > &p_list) {
	glm::vec3 fext = glm::vec3(0.0f);
	unsigned int num_neighbors = p.notRigidBodyNeighbours.size();
	
	#pragma omp parallel for
	for (int k = 0; k < num_neighbors; k++) {
		float kij = (2 * REST) / (p.rho + p_list[p.notRigidBodyNeighbours[k]].rho);
		Particle currentng = p_list[p.notRigidBodyNeighbours[k]];
		glm::vec3 cohesionNow = cohesion(p, p_list[p.notRigidBodyNeighbours[k]]);
		glm::vec3 curvatureNow = curvature(p, p_list[p.notRigidBodyNeighbours[k]], p_list);
		fext = fext + (kij * (cohesionNow + curvatureNow));
	}
	return fext;

}

//Caculates delta p
void CalculateDp(std::vector<Particle> &predict_p) {

	float dqMag = g_dq * g_h;
	float kpoly = 315.0f / (64.0f * PI * pow(g_h, 9));
	float wQH = kpoly * pow((g_h * g_h - dqMag * dqMag), 3);

	unsigned int num_particles = predict_p.size();

	//For each particle
	#pragma omp parallel for 
	for (int i = 0; i < num_particles; i++) {

		//If particle isnt rigidBody or colliding with rigidBody
		if (!predict_p[i].isRigidBody && !predict_p[i].isCollidingWithRigidBody) {

			int neighborsize = predict_p[i].allNeighbours.size();

			glm::vec3 res = glm::vec3(0.0f);
			glm::vec3 r;

			//For each neighbour of that particle
			#pragma omp parallel for
			for (int j = 0; j < neighborsize; j++) {

				//Gets distance of particle to neighbour
				r = predict_p[i].current_position - predict_p[predict_p[i].allNeighbours[j]].current_position;

				//Sums particle and neighbour lambda
				float lambdaSum = predict_p[i].lambda + predict_p[predict_p[i].allNeighbours[j]].lambda;

				//Applies kernel an accumulates multiplied with lambda sum
				res += (lambdaSum /*+ scorr*/)* wSpiky(r, g_h);
			}

			
			predict_p[i].delta_p = res / REST;
		}
	}
}

//ETA
glm::vec3 eta(Particle &p, std::vector<Particle> &predict_p, float &vorticityMag) {
	glm::vec3 eta = glm::vec3(0.0f);
	int neighborsize = p.allNeighbours.size();
	if (neighborsize > 0) {
		#pragma omp parallel for
		for (int j = 0; j < neighborsize; j++) {
			glm::vec3 r = p.current_position - predict_p[p.allNeighbours[j]].current_position;
			eta += wSpiky(r, g_h) * vorticityMag;
		}
	}

	return eta;
}

//Calculates vorticity
glm::vec3 VorticityConfinement(Particle &p, std::vector< Particle > &p_list) {
	glm::vec3 omega = glm::vec3(0.0f);
	int num_neighbors = p.allNeighbours.size();

	#pragma omp parallel for
	for (int k = 0; k < num_neighbors; k++) {
		glm::vec3 v_ij = p_list[p.allNeighbours[k]].velocity - p.velocity;
		glm::vec3 r = p.current_position - p_list[p.allNeighbours[k]].current_position;
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

//Calculates viscosity
glm::vec3 XSPHViscosity(Particle &p, std::vector< Particle > &p_list)
{
	unsigned int num_neighbors = p.allNeighbours.size();
	glm::vec3 visc = glm::vec3(0.0f);
	if (num_neighbors > 0) {
		#pragma omp parallel for
		for (int k = 0; k < num_neighbors; k++)
		{
			glm::vec3 v_ij = p_list[p.allNeighbours[k]].velocity - p.velocity;
			glm::vec3 r = p.current_position - p_list[p.allNeighbours[k]].current_position;
			visc += v_ij * wPoly6(r, g_h);
		}
	}
	return visc * viscosityC;
}

//Collision Detection and Response
void CollisionDetectionResponse(std::vector< Particle > &p_list)
{
	unsigned int num_particles = p_list.size();

	//For each particle
	#pragma omp parallel for
	for (int i = 0; i < num_particles; i++) {

		//If it's is colliding with min Z boundary
		if (predict_p[i].current_position.z < g_zmin + boundary)
			predict_p[i].current_position.z = g_zmin + boundary;
		

		//If it's is colliding with max Z boundary
		if (predict_p[i].current_position.z > g_zmax - boundary) 
			predict_p[i].current_position.z = g_zmax - boundary;

		//If it's is colliding with min Y boundary
		if (predict_p[i].current_position.y < g_ymin + boundary) 
			predict_p[i].current_position.y = g_ymin + boundary;

		//If it's is colliding with min X boundary
		if (predict_p[i].current_position.x < g_xmin + boundary)
			predict_p[i].current_position.x = g_xmin + boundary;

		//If it's is colliding with max X boundary'
		if (predict_p[i].current_position.x > g_xmax - boundary) 
			predict_p[i].current_position.x = g_xmax - boundary;

	}
}


//Hash Function - LEGACY (suboptimal)
int ComputeHash(int &grid_x, int &grid_y, int &grid_z)
{
	int grid = (grid_x + grid_y * GRID_RESOLUTION) + grid_z * (GRID_RESOLUTION * GRID_RESOLUTION);

	return grid;
}

//------------------------------------------------------------------------------

//Creates Hash Table - LEGACY (Suboptimal)
void BuildHashTable(std::vector<Particle> &p_list, Hash &hash_table)
{
	int num_particles = p_list.size();
	int grid_x;
	int grid_y;
	int grid_z;

	float cell_size = (g_xmax - g_xmin) / GRID_RESOLUTION;
	/*printf("Grid: %f\n", cell_size);*/

	hash_table.clear();

	#pragma omp parallel for
	for (int i = 0; i < num_particles; i++)
	{
		grid_x = floor(p_list[i].current_position[0] / cell_size);
		grid_y = floor(p_list[i].current_position[1] / cell_size);
		grid_z = floor(p_list[i].current_position[2] / cell_size);

		p_list[i].hash = ComputeHash(grid_x, grid_y, grid_z);

		hash_table.insert(Hash::value_type(p_list[i].hash, i));
		/*printf("Grid:\n");*/

	}
}

//------------------------------------------------------------------------------

//Creates neighborslist - LEGACY (really suboptimal and current bottleneck)
void SetUpNeighborsLists(std::vector<Particle> &p_list, Hash &hash_table)
{
	#pragma omp parallel
	{
		int num_particles = p_list.size();

		float cell_size = (g_xmax - g_xmin) / GRID_RESOLUTION;

		int x_idx;
		int y_idx;
		int z_idx;

		//Min-Max index in X axis
		int grid_x_max;
		int grid_x_min;

		//Min-Max index in Y axis
		int grid_y_max;
		int grid_y_min;

		//Min-Max index in Z axis
		int grid_z_max;
		int grid_z_min;

		//Hash function result
		int hash;


		#pragma omp parallel for nowait
		//For all particles
		for (int i = 0; i < num_particles; i++) {

			//Clear particles neighbours
			p_list[i].allNeighbours.clear();

			//Calculates min-max grid index based on current particle position
			grid_x_min = floor((p_list[i].current_position[0] - g_h) / cell_size);
			grid_y_min = floor((p_list[i].current_position[1] - g_h) / cell_size);
			grid_z_min = floor((p_list[i].current_position[2] - g_h) / cell_size);

			grid_x_max = floor((p_list[i].current_position[0] + g_h) / cell_size);
			grid_y_max = floor((p_list[i].current_position[1] + g_h) / cell_size);
			grid_z_max = floor((p_list[i].current_position[2] + g_h) / cell_size);

			//Iterate from min Z index to max Z index
			#pragma omp parallel for nowait
			for (z_idx = grid_z_min; z_idx <= grid_z_max; z_idx++) {
				//Iterate from min Y index to max Y index
				#pragma omp parallel for nowait
				for (y_idx = grid_y_min; y_idx <= grid_y_max; y_idx++) {
					//Iterate from min X index to max X index
					#pragma omp parallel for nowait
					for (x_idx = grid_x_min; x_idx <= grid_x_max; x_idx++) {

							//Get Hash result for current X,Y,Z index
							hash = ComputeHash(x_idx, y_idx, z_idx);
						
							//Get Pair with range (starting, end) of values with this hash index
							auto its = hash_table.equal_range(hash);

							//Iterate from starting value to end
							#pragma omp parallel for 
							for (auto it = its.first; it != its.second; ++it) {

								//Avoid comparing with itself
								if (it->second != i) {

									//If current particles are closer than g_h factor  
									if (length(p_list[i].current_position - p_list[it->second].current_position) <= g_h) {

										//If current particle and current neighbour are rigidBody
										if (p_list[i].isRigidBody && p_list[it->second].isRigidBody)
											p_list[i].rigidBodyNeighbours.push_back(it->second);	//Push neigbhour to rigidBody vector

										//If current particle not rigidBody, but neighbour is
										if (!p_list[i].isRigidBody && p_list[it->second].isRigidBody) {
											p_list[i].rigidBodyNeighbours.push_back(it->second);	//Push neighbour to rigidBody vector

											//If particle is closer to rigidBody than wall_h factor
											if (glm::length(p_list[i].current_position - p_list[it->second].current_position) < wall_h)
												p_list[i].isCollidingWithRigidBody = true;	//Set isCollidingWithRigidBody constant to true

										}

										//If current particle is rigidBody, and neighbour is isCollidingWithRigidBody
										if (p_list[i].isRigidBody && p_list[it->second].isCollidingWithRigidBody)
											p_list[i].rigidBodyNeighbours.push_back(it->second);	//Push neighbour to rigidBody vector

										//If both particles are not rigidBody
										if (!p_list[i].isRigidBody && !p_list[it->second].isRigidBody)	
											p_list[i].notRigidBodyNeighbours.push_back(it->second);	//Push to all vector

										//Push to neighbors vector
										p_list[i].allNeighbours.push_back(it->second);

									}

								}
							}
					}
				}
			}

		}
	}
}

//Calculates boundary volume
float boundaryVolume(unsigned int &i, std::vector< Particle > &p_list) {
	float massboundary = 0;
	glm::vec3 r;

	int neighborsize = p_list[i].rigidBodyNeighbours.size();
	#pragma omp parallel for
	for (int j = 0; j < neighborsize; j++) {
		glm::vec3 r = p_list[i].current_position - p_list[p_list[i].rigidBodyNeighbours[j]].current_position;
		massboundary += wPoly6(r, h_adhesion);
	}
	massboundary = 1 / massboundary;
	massboundary = REST*massboundary;
	return massboundary;
}

//Calculates surface adhesion
glm::vec3 adhesion(Particle &p, std::vector< Particle > &p_list) {
	glm::vec3 fext = glm::vec3(0.0f);
	unsigned int num_neighbors = p.rigidBodyNeighbours.size();
	float term = 0.007 / pow(h_adhesion, 3.25f);
	#pragma omp parallel for
	for (int k = 0; k < num_neighbors; k++) {
		glm::vec3 r = p.current_position - p_list[p.rigidBodyNeighbours[k]].current_position;
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
		float boundaryVol = boundaryVolume(p.rigidBodyNeighbours[k], p_list);
		float coeff = adhesioncoeff;
		if (p_list[p.rigidBodyNeighbours[k]].pencil)
			coeff = adhesioncoeff;
		fext = fext + ((-coeff)*(p.mass)*(boundaryVol)*(spline)*(divR));
	}

	return fext;

}

//Calculates particle friction
glm::vec3 particleFriction(Particle &p, std::vector< Particle > &p_list, int i) {
	glm::vec3 deltax = glm::vec3(0.0f);
	unsigned int num_neighbors = p.rigidBodyNeighbours.size();
	#pragma omp parallel for
	for (int k = 0; k < 1; k++) {
		glm::vec3 r = p.current_position - p_list[p.rigidBodyNeighbours[k]].current_position;
		float distance = length(r) - g_h;
		glm::vec3 n = r / length(r);
		glm::vec3 xi = particlesList[i].current_position - p.current_position;
		glm::vec3 xj = particlesList[p.rigidBodyNeighbours[k]].current_position - p_list[p.rigidBodyNeighbours[k]].current_position;
		glm::vec3 perp = glm::perp((xi - xj), n);

		float invmass = 1 / p.mass;
		invmass = invmass / (invmass + invmass);
		if (length(perp) < (stattc * distance)) 
			deltax += invmass * perp;
		else {
			float mini = min((kinetic*distance) / length(perp), 1.0f);
			deltax += invmass * (perp*mini);
		}

		predict_p[p.rigidBodyNeighbours[k]].velocity = -invmass * deltax;

	}

	return deltax;

}

void movewallz(std::vector<Particle> &p_list) {
	int num_particles = p_list.size();

	for (int i = 0; i < num_particles; i++) {
		if (p_list[i].pencil)
			p_list[i].current_position.z = p_list[i].current_position.z + positions.z;
	}
}

void movewallx(std::vector<Particle> &p_list) {
	int num_particles = p_list.size();

	for (int i = 0; i < num_particles; i++) {
		if (p_list[i].pencil) 
			p_list[i].current_position = (positions + glm::vec3(p_list[i].varx, p_list[i].vary, -1.0f));
	}
}

void movewally(std::vector<Particle> &p_list) {
	int num_particles = p_list.size();

	for (int i = 0; i < num_particles; i++) {
		if (p_list[i].pencil) 
			p_list[i].current_position.y = p_list[i].current_position.y + move_wally;
	}
}