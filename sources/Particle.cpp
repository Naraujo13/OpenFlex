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

/* -- Particle Related Functions -- */
/*
//Initialize particle list
void InitializeParticleList()
{
	//Clear particles vector
	particlesList.clear();
	
	//Start positioning particles at some distance from the left and bottom walls
	float x_ini_pos = g_xmax / 2.0f - boundary;
	float y_ini_pos = g_ymin + boundary;
	float z_ini_pos = g_zmax / 2 - boundary;

	//Deltas for particle distribution
	float d_x = 0.056f;
	float d_y = 0.056f;
	float d_z = 0.056f;

	std::cout << "Initializing " <<  (PARTICLE_COUNT_X * PARTICLE_COUNT_Y * PARTICLE_COUNT_Z) << " particles..." << std::endl;

	float x_pos = x_ini_pos;

	for (unsigned int x = 0; x < PARTICLE_COUNT_X; x++){
		float y_pos = y_ini_pos;

		for (unsigned int y = 0; y < PARTICLE_COUNT_Y; y++){
			float z_pos = z_ini_pos;

			for (unsigned int z = 0; z < PARTICLE_COUNT_Z; z++){
				
				//Gets random number and normalize (0-1)
				float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

				//Puts new particle in starting position + random, with standard constraints
				Particle p(
					glm::vec3(x_pos + r, y_pos + r, z_pos + r),
					glm::vec3(0.0f),
					glm::vec3(0.0f),
					glm::vec3(0.0f),
					1.0f,
					0.0f,
					0.0f,
					1.0f,
					0.0f,
					false,
					false,
					false,
					false

				);			

				//Pushes to vector
				particlesList.push_back(p);
				z_pos += d_z;
			}
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	predictionList = particlesList;
}
*/

/* -- PBD Kernel Functions -- */
/*
//Kernel for Density Estimation
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

//Kernel for Gradient Calculator
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
*/


/* --- Spatial Hash --- */

//Clean
void SpatialHash::cleanHash() {
	this->cells.clear(); 
	std::cout << "\t\t" << "Hash cleaned. Size: " << cells.size() << std::endl;
}

void SpatialHash::buildHash(glm::vec3 dim, float cellSize) {
	


	this->cellSize = cellSize;
	glm::vec3 size;
	size.x = ceil(dim.x / cellSize);
	size.y = ceil(dim.y / cellSize);
	size.z = ceil(dim.z / cellSize);
	this->dim = size;

	//Debug
	//std::cout << "\tTamanho de uma célula: " << cellSize << std::endl;
	//std::cout << "\tTamanho em células da spatial hash: " << this->dim.x << ", " << this->dim.y << ", " << this->dim.z << std::endl;
	//getchar();

	#pragma omp parallel for
	for (int x = 0; x < dim.x; x++) {
		#pragma omp parallel for
		for (int y = 0; y < dim.y; y++) {
			#pragma omp parallel for
			for (int z = 0; z < dim.z; z++) {
				cells.insert(spatialCell::value_type(glm::vec3(x, y, z), particleVector()));
			}
		}
	}
}

//Construtor
SpatialHash::SpatialHash() {}

SpatialHash::SpatialHash(glm::vec3 dim, float cellSize) {
	this->cellSize = cellSize;
	glm::vec3 size;
	size.x = ceil(dim.x / cellSize);
	size.y = ceil(dim.y / cellSize);
	size.z = ceil(dim.z / cellSize);
	this->dim = size;
	buildHash(this->dim, cellSize);
}

void SpatialHash::reconstruct() {
	this->cleanHash();
	this->buildHash(this->dim, this->cellSize);
}

void SpatialHash::reconstruct(glm::vec3 dim, float cellSize) {
	this->cleanHash();
	this->buildHash(dim, cellSize);
}

//Hash
glm::vec3 SpatialHash::hashFunction(glm::vec3 particlePosition) {

	//Grid Id's
	int grid_id_x;
	int grid_id_y;
	int grid_id_z;

	//Get cell x
	grid_id_x = (floor(particlePosition.x / this->cellSize));

	//Get cell y
	grid_id_y = (floor(particlePosition.y / this->cellSize));

	//Get cell z
	grid_id_z = (floor(particlePosition.z / this->cellSize));

	//Debug
	//std::cout << "\t\tHashing: (" << particlePosition.x << ", " << particlePosition.y << ", " << particlePosition.z << ") -> (";
	//std::cout << grid_id_x << ", " << grid_id_y << ", " << grid_id_z << ")" << std::endl;
	//getchar();

	return glm::vec3(grid_id_x, grid_id_y, grid_id_z);

}

//Insert
bool SpatialHash::insert(glm::vec3 particlePos, int particleIndex) {

	//std::cout << "\tGetting hash value...\n";
	glm::vec3 index = this->hashFunction(particlePos);

	//std::cout << "\tInserting at (" << index.x << ", " << index.y << ", " << index.z << ")\n";
	
	//std::cout << "\tVector size before: " << this->cells[index].size() << "\n"; 
		
	this->cells[index].push_back(particleVector::value_type(particleIndex));
	
	//std::cout << "\tVector size after: " << this->cells[index].size() << "\n";

	//getchar();
	

	return true;
}

//Get cell
std::vector<int> SpatialHash::getCell(glm::vec3 particlePosition) {
	//glm::vec3 index = hashFunction(particlePosition);
	return this->cells[particlePosition];
}