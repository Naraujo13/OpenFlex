#include "Particle.hpp"

/* -- Variables -- */

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

/* -- Particle Functions -- */
Particle::Particle(
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
	bool hybrid)
{
	this->position = startPosition;
	this->pred_position = pred_position;
	this->velocity = velocity;
	this->delta_p = delta_p;
	this->mass = mass;
	this->lambda = lambda;
	this->rho = rho;
	this->C = C;
	this->phase = phase;
	this->teardrop = teardrop;
	this->wall = wall;
	this->pencil = pencil;
	this->hybrid = hybrid;

}

/* -- Particle Related Functions -- */

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

/* -- Updates neighbours -- */

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
			p_list[i].neighbors.clear();

			//Calculates min-max grid index based on current particle position
			grid_x_min = floor((p_list[i].position[0] - g_h) / cell_size);
			grid_y_min = floor((p_list[i].position[1] - g_h) / cell_size);
			grid_z_min = floor((p_list[i].position[2] - g_h) / cell_size);

			grid_x_max = floor((p_list[i].position[0] + g_h) / cell_size);
			grid_y_max = floor((p_list[i].position[1] + g_h) / cell_size);
			grid_z_max = floor((p_list[i].position[2] + g_h) / cell_size);

			//Iterate from min Z index to max Z index
			for (z_idx = grid_z_min; z_idx <= grid_z_max; z_idx++) {
				//Iterate from min Y index to max Y index
				for (y_idx = grid_y_min; y_idx <= grid_y_max; y_idx++) {
					//Iterate from min X index to max X index
					for (x_idx = grid_x_min; x_idx <= grid_x_max; x_idx++) {

						//Get Hash result for current X,Y,Z index
						hash = ComputeHash(x_idx, y_idx, z_idx);

						//Get Pair with range (starting, end) of values with this hash index
						auto its = hash_table.equal_range(hash);

						//Iterate from starting value to end
						for (auto it = its.first; it != its.second; ++it) {

							//Avoid comparing with itself
							if (it->second != i) {

								//If current particles are closer than g_h factor  
								if (length(p_list[i].position - p_list[it->second].position) <= g_h) {

									//If current particle and current neighbour are wall
									if (p_list[i].wall && p_list[it->second].wall)
										p_list[i].wneighbors.push_back(it->second);	//Push neigbhour to wall vector

																					//If current particle not wall, but neighbour is
									if (!p_list[i].wall && p_list[it->second].wall) {
										p_list[i].wneighbors.push_back(it->second);	//Push neighbour to wall vector

																					//If particle is closer to wall than wall_h factor
										if (glm::length(p_list[i].position - p_list[it->second].position) < wall_h)
											p_list[i].hybrid = true;	//Set hybrid constant to true

									}

									//If current particle is wall, and neighbour is hybrid
									if (p_list[i].wall && p_list[it->second].hybrid)
										p_list[i].wneighbors.push_back(it->second);	//Push neighbour to wall vector

																					//If both particles are not wall
									if (!p_list[i].wall && !p_list[it->second].wall)
										p_list[i].allneighbors.push_back(it->second);	//Push to all vector

																						//Push to neighbors vector
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


/* -- PBD Kernel Functions -- */

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