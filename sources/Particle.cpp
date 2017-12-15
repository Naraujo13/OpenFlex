#include "particle.hpp"

/* -- Variables -- */
/*
//Defines
//Stating number of particles
#define PARTICLE_COUNT_X 10
#define PARTICLE_COUNT_Y 2
#define PARTICLE_COUNT_Z 10

//Particles
std::vector<Particle> particlesList;

//????
std::vector< Particle > predictionList;

//Cell boundaries
#define g_xmax 2
#define g_xmin 0
#define g_ymax 2
#define g_ymin 0
#define g_zmax 2
#define g_zmin 0

//Wall boundary
#define boundary 0.03f
*/
/* -- Particle Functions -- */

//Construtor
Particle::Particle() {}

Particle::Particle(
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
	bool isCollidingWithRigidBody)
{
	this->current_position = startPosition;
	this->predicted_position = predicted_position;
	this->velocity = velocity;
	this->delta_p = delta_p;
	this->mass = mass;
	this->lambda = lambda;
	this->rho = rho;
	this->C = C;
	this->phase = phase;
	this->teardrop = teardrop;
	this->isRigidBody = isRigidBody;
	this->pencil = pencil;
	this->isCollidingWithRigidBody = isCollidingWithRigidBody;

}