
#include "pbf.hpp"

Hash hash_table;


std::vector<ParticleStruct> particlesList;
std::vector<ParticleStruct> predict_p;
//std::vector<float> g_grid;
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

//Initialize particle list
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
				p.C = 0;

				p.predicted_position.x = 0;
				p.predicted_position.y = 0;
				p.predicted_position.z = 0;

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

//Creates rigid body cube
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

			particlesList.push_back(p);
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

			particlesList.push_back(p);
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

			particlesList.push_back(p);
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

			particlesList.push_back(p);
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

			particlesList.push_back(p);
			y_pos += d_y;
		}
		x_pos += d_x;
	}

	printf("Number of particles in the simulation: %i.\n", particlesList.size());

	predict_p = particlesList;
}

//Spawn new fluid particles
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
				ParticleStruct p;

				float r = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) / 100.0f;

				//float v = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) -0.5f) / 100.0f;


				p.current_position.x = x_pos;
				p.current_position.y = y_pos;
				p.current_position.z = z_pos;

				p.velocity.x = 0;
				p.velocity.y = 0;
				p.velocity.z = -2.0f;
				
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

//Estimate rho for particle i - updated to opencl, needs testing
void DensityEstimator(std::vector<ParticleStruct> &predict_p, std::vector<int> neighborBins, int* binBoundaries, int &i) {

	float rhof = 0.0f;
	float rhos = 0.0f;
	glm::vec3 r;

	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//Gets distance to neighbour
			r = glm::vec3(
				predict_p[i].current_position.x - neighborParticle.current_position.x,
				predict_p[i].current_position.y - neighborParticle.current_position.y,
				predict_p[i].current_position.z - neighborParticle.current_position.z
			);

			//If neighbor is NOT a rigid body, accumulate smoothing kernel(delta, radius) in rhof
			if (!neighborParticle.isRigidBody)	
				rhof += wPoly6(r, g_h);
			else 	//If it IS a rigid body, accumulate smoothing kernel(delta, radius) in rhos
				rhos += wPoly6(r, g_h);
			

			//Set rho for analyzed particle
			predict_p[i].rho = rhof + (solid * rhos);
			
		}
	}

}

//Nabla - ported to opencl, needs extensive testing
float NablaCSquaredSumFunction(ParticleStruct &p, std::vector<ParticleStruct> &predict_p, std::vector<int> neighborBins, int* binBoundaries) {
	
	glm::vec3 r;
	std::vector<glm::vec3> NablaC;
	float res = 0.0f;

	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF

			//Gets distance to neighbour
			r = glm::vec3(
				p.current_position.x - neighborParticle.current_position.x,
				p.current_position.y - neighborParticle.current_position.y,
				p.current_position.z - neighborParticle.current_position.z
			);

			//Applies kernel and divides by rest?
			glm::vec3 nablac = -wSpiky(r, g_h) / REST;

			//Push to NablaC vector
			NablaC.push_back(nablac);
		}
	}

	//Legacy (why do this?) TODO: Discover why this
	NablaC.push_back(glm::vec3(0.0f));


	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF

			//Gets distance to neighbour
			r = glm::vec3(
				p.current_position.x - neighborParticle.current_position.x,
				p.current_position.y - neighborParticle.current_position.y,
				p.current_position.z - neighborParticle.current_position.z
			);

			//Sums distance in the last position of Nabla vector
			int last = NablaC.size() - 1;
			NablaC[last] = NablaC[last] + wSpiky(r, g_h);

			//Divides last position for REST
			NablaC[last] = NablaC[last] / REST;
		}
	}

	//For each element in Nabla vector 
	#pragma omp parallel for
	for (int k = 0; k < NablaC.size(); k++) {
		float norm_nablac = length(NablaC[k]);
		res += norm_nablac * norm_nablac;
	}
	return res;
}

glm::vec3 cohesion(ParticleStruct &p, ParticleStruct &p_neighbor) {

	glm::vec3 fext = glm::vec3(0.0f);
	float term = 32 / (PI * pow(g_h, 9));

	glm::vec3 r(
		p.current_position.x - p_neighbor.current_position.x,
		p.current_position.y - p_neighbor.current_position.y,
		p.current_position.z - p_neighbor.current_position.z
	);

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

//Ported to opencl, needs revision
glm::vec3 surfaceArea(ParticleStruct &p, std::vector <int> neighborBins, cl_int* h_binBoundaries) {
	glm::vec3 fext = glm::vec3(0.0f);
	
	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = h_binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = h_binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF

			//If neighbor is NOT rigid body
			if (!neighborParticle.isRigidBody) {
				glm::vec3 r (
					p.current_position.x - neighborParticle.current_position.x,
					p.current_position.y - neighborParticle.current_position.y,
					p.current_position.z - neighborParticle.current_position.z
				);

				float spline = neighborParticle.mass / neighborParticle.rho;
				fext = fext + (spline * wSpiky(r, g_h));

			}
		}
	}

	return g_h * fext;

}

//Ported to opencl, needs revision
glm::vec3 curvature(ParticleStruct &p, ParticleStruct &p_neighbor, std::vector <int> neighborBins, cl_int* binBoundaries, cl_int* numBins) {
	glm::vec3 fext = glm::vec3(0.0f);

	glm::vec3 ni = surfaceArea(p, neighborBins, binBoundaries);
	glm::vec3 nj = surfaceArea(p_neighbor, getNeighbourBins(p_neighbor.hash, numBins, numBins[0] * numBins[1] * numBins[2]), binBoundaries);
	fext = fext + (-cohesioncoeff*p.mass*(ni - nj));
	return fext;

}

//Ported to opencl, needs revision
glm::vec3 surfaceTension(ParticleStruct &p, std::vector <int> neighborBins, cl_int* binBoundaries, cl_int* numBins) {

	glm::vec3 fext = glm::vec3(0.0f);

	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF

			//if neighbor is NOT a rigid body
			if (!neighborParticle.isRigidBody) {
				float kij = (2 * REST) / (p.rho + neighborParticle.rho);
				glm::vec3 cohesionNow = cohesion(p, neighborParticle);
				glm::vec3 curvatureNow = 
					curvature(
						p, 
						neighborParticle, 
						neighborBins,
						binBoundaries,
						numBins);
				fext = fext + (kij * (cohesionNow + curvatureNow));
			}
		}
	}

	return fext;

}


//Ported to opencl, needs revision
void CalculateDp(std::vector<ParticleStruct> &predict_p, cl_int* numBins, cl_int* binBoundaries) {

	float dqMag = g_dq * g_h;
	float kpoly = 315.0f / (64.0f * PI * pow(g_h, 9));
	float wQH = kpoly * pow((g_h * g_h - dqMag * dqMag), 3);

	unsigned int num_particles = predict_p.size();

	//For each particle
	#pragma omp parallel for 
	for (int i = 0; i < num_particles; i++) {

		//Get Neighbouring Bins REMOVE
		std::vector <int> neighborBins(getNeighbourBins(predict_p[i].hash, numBins, numBins[0] * numBins[1] * numBins[2]));

		//If particle isnt rigidBody or colliding with rigidBody
		if (!predict_p[i].isRigidBody && !predict_p[i].isCollidingWithRigidBody) {

			glm::vec3 res = glm::vec3(0.0f);
			glm::vec3 r;

			//For each neighboring bin
			for (int index = 0; index < neighborBins.size(); index++) {

				//Get binStart and binEnd
				int binStart = binBoundaries[index];
				int binEnd;
				if (index < neighborBins.size() - 1)
					binEnd = binBoundaries[index + 1];
				else
					binEnd = neighborBins.size();

				//For each particle in that bin
				for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

					//Gets Neighbour
					ParticleStruct neighborParticle = predict_p[particleIndex];

					//DO NEIGHBOUR STUFF	
					r = glm::vec3(
						predict_p[i].current_position.x - neighborParticle.current_position.x,
						predict_p[i].current_position.y - neighborParticle.current_position.y,
						predict_p[i].current_position.z - neighborParticle.current_position.z
					);

					//Sums particle and neighbour lambda
					float lambdaSum = predict_p[i].lambda + neighborParticle.lambda;

					//Applies kernel an accumulates multiplied with lambda sum
					res += (lambdaSum /*+ scorr*/)* wSpiky(r, g_h);

				}
			}
			predict_p[i].delta_p.x = res.x / REST;
			predict_p[i].delta_p.y = res.y / REST;
			predict_p[i].delta_p.z = res.z / REST;
		}
	}
}

//Ported to opencl, needs revision
glm::vec3 eta(ParticleStruct &p, float &vorticityMag, std::vector<int> neighborBins, cl_int* binBoundaries) {
	glm::vec3 eta = glm::vec3(0.0f);

	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that bin
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF	
			glm::vec3 r(
				p.current_position.x - neighborParticle.current_position.x,
				p.current_position.y - neighborParticle.current_position.y,
				p.current_position.z - neighborParticle.current_position.z
			);
			eta += wSpiky(r, g_h) * vorticityMag;
		}
	}
	
	return eta;
}

//Ported to opencl, needs revision
glm::vec3 VorticityConfinement(ParticleStruct &p, std::vector<int> neighborBins, cl_int* binBoundaries) {
	
	glm::vec3 omega = glm::vec3(0.0f);

	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that bin
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF	

			glm::vec3 v_ij(
				neighborParticle.velocity.x - p.velocity.x,
				neighborParticle.velocity.y - p.velocity.y,
				neighborParticle.velocity.z - p.velocity.z
			);

			glm::vec3 r(
				p.current_position.x - neighborParticle.current_position.x,
				p.current_position.y - neighborParticle.current_position.y,
				p.current_position.z - neighborParticle.current_position.z
			);

			omega += glm::cross(v_ij, wSpiky(r, g_h));

		}
	}

	
	float omegaLength = glm::length(omega);
	if (omegaLength == 0.0f) //No direction for eta
		return glm::vec3(0.0f);
	

	glm::vec3 etaVal = eta(p, omegaLength, neighborBins, binBoundaries);

	if (etaVal.x == 0 && etaVal.y == 0 && etaVal.z == 0) //Particle is isolated or net force is 0
		return glm::vec3(0.0f);
	
	glm::vec3 n = normalize(etaVal);

	return (glm::cross(n, omega) * vorticityEps);
}

//Ported to opencl, needs revision
glm::vec3 XSPHViscosity(ParticleStruct &p, std::vector<int> neighborBins, cl_int* binBoundaries)
{

	glm::vec3 visc = glm::vec3(0.0f);

	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that bin
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF	

			glm::vec3 v_ij (
				neighborParticle.velocity.x - p.velocity.x,
				neighborParticle.velocity.y - p.velocity.y,
				neighborParticle.velocity.z - p.velocity.z
			);

			glm::vec3 r(
				p.current_position.x - neighborParticle.current_position.x,
				p.current_position.y - neighborParticle.current_position.y,
				p.current_position.z - neighborParticle.current_position.z
			);

			visc += v_ij * wPoly6(r, g_h);

		}
	}

	return visc * viscosityC;
}

//Collision Detection and Response
void CollisionDetectionResponse(std::vector< ParticleStruct > &p_list)
{
	unsigned int num_particles = p_list.size();

	/*glm::vec2 w_min(wall_min_x, wall_min_y);
	glm::vec2 w_max(wall_max_x, wall_max_y);*/

	//For each particle
	#pragma omp parallel for
	for (int i = 0; i < num_particles; i++) {

		//If it's is colliding with min Z boundary
		if (predict_p[i].current_position.z < g_zmin + boundary) {
			predict_p[i].current_position.z = g_zmin + boundary;

			/*glm::vec3 normal = glm::vec3(0,0,1);
			predict_p[i].velocity.z = glm::reflect(predict_p[i].velocity, normal).z * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/

		}

		//If it's is colliding with max Z boundary
		if (predict_p[i].current_position.z > g_zmax - boundary) {
			predict_p[i].current_position.z = g_zmax - boundary;

			/*glm::vec3 normal = glm::vec3(0,0,-1);
			predict_p[i].velocity.z = glm::reflect(predict_p[i].velocity, normal).z * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/

		}
		
		//If it's is colliding with min Y boundary
		if (predict_p[i].current_position.y < g_ymin + boundary) {
			predict_p[i].current_position.y = g_ymin + boundary;

			/*glm::vec3 normal = glm::vec3(0,1,0);
			predict_p[i].velocity.y = glm::reflect(predict_p[i].velocity, normal).y * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/

		}

		//If it's is colliding with min X boundary
		if (predict_p[i].current_position.x < g_xmin + boundary) {
			predict_p[i].current_position.x = g_xmin + boundary;

			/*glm::vec3 normal = glm::vec3(1,0,0);
			predict_p[i].velocity.x = glm::reflect(predict_p[i].velocity, normal).x * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/

		}

		//If it's is colliding with max X boundary
		if (predict_p[i].current_position.x > g_xmax - boundary) {
			predict_p[i].current_position.x = g_xmax - boundary;

			/*glm::vec3 normal = glm::vec3(-1,0,0);
			predict_p[i].velocity.x = glm::reflect(predict_p[i].velocity, normal).x * DT;*/
			/*predict_p[i].position = particles[i].position + predict_p[i].velocity * BOUNCE;*/

		}
	}
}

//Ported to opencl, needs revision
float boundaryVolume(ParticleStruct p, std::vector<int> neighborBins, cl_int* binBoundaries) {

	float massboundary = 0;
	glm::vec3 r;

	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF	
		
			//If neighbor IS rigid body
			if (neighborParticle.isRigidBody) {

				//Gets distance to neighbour
				glm::vec3 r(
					p.current_position.x - neighborParticle.current_position.x,
					p.current_position.y - neighborParticle.current_position.y,
					p.current_position.z - neighborParticle.current_position.z
				);

				//Accumulates mass boundary with Poly6 kernel
				massboundary += wPoly6(r, h_adhesion);
			}
		}
	}

	//Normalize
	massboundary = 1 / massboundary;
	massboundary = REST * massboundary;

	return massboundary;
}

//Ported to opencl, needs revision
glm::vec3 adhesion(ParticleStruct &p, std::vector <int> neighborBins, cl_int* h_binBoundaries) {

	glm::vec3 fext = glm::vec3(0.0f);
	

	float term = 0.007 / pow(h_adhesion, 3.25f);


	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = h_binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = h_binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF

			//If neighbour IS rigid body
			if (neighborParticle.isRigidBody){

				//Gets distance to neighbour
				glm::vec3 r = glm::vec3(
					p.current_position.x - neighborParticle.current_position.x,
					p.current_position.y - neighborParticle.current_position.y,
					p.current_position.z - neighborParticle.current_position.z
				);

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

				float boundaryVol = boundaryVolume(neighborParticle, neighborBins, h_binBoundaries);

				float coeff = adhesioncoeff;

				if (neighborParticle.pencil)
					coeff = adhesioncoeff;

				fext = fext + ((-coeff)*(p.mass)*(boundaryVol)*(spline)*(divR));

			}

		}
	}

	return fext;

}

//Ported to opencl, needs revision
glm::vec3 particleFriction(ParticleStruct &p, std::vector <int> neighborBins, cl_int* h_binBoundaries, int i) {

	glm::vec3 deltax = glm::vec3(0.0f);
	
	//For each neighboring bin
	for (int index = 0; index < neighborBins.size(); index++) {

		//Get binStart and binEnd
		int binStart = h_binBoundaries[index];
		int binEnd;
		if (index < neighborBins.size() - 1)
			binEnd = h_binBoundaries[index + 1];
		else
			binEnd = neighborBins.size();

		//For each particle in that
		for (int particleIndex = binStart; particleIndex < binEnd; particleIndex++) {

			//Gets Neighbour
			ParticleStruct neighborParticle = predict_p[particleIndex];

			//DO NEIGHBOUR STUFF

			//If neighbor IS rigid body
			if (neighborParticle.isRigidBody) {

				//Gets distance to neighbour
				glm::vec3 r = glm::vec3(
					p.current_position.x - neighborParticle.current_position.x,
					p.current_position.y - neighborParticle.current_position.y,
					p.current_position.z - neighborParticle.current_position.z
				);

				float distance = length(r) - g_h;
				
				glm::vec3 n = r / length(r);
				
				glm::vec3 xi(
					particlesList[i].current_position.x - p.current_position.x,
					particlesList[i].current_position.y - p.current_position.x,
					particlesList[i].current_position.z - p.current_position.z
				);

				glm::vec3 xj(
					particlesList[particleIndex].current_position.x - neighborParticle.current_position.x,
					particlesList[particleIndex].current_position.y - neighborParticle.current_position.y,
					particlesList[particleIndex].current_position.z - neighborParticle.current_position.z
				);


				glm::vec3 perp = glm::perp((xi - xj), n);

				float invmass = 1 / p.mass;
				invmass = invmass / (invmass + invmass);
				if (length(perp) < (stattc * distance))
					deltax += invmass * perp;
				else {
					float mini = min((kinetic*distance) / length(perp), 1.0f);
					deltax += invmass * (perp*mini);
				}

				neighborParticle.velocity.x = -invmass * deltax.x;
				neighborParticle.velocity.y= -invmass * deltax.y;
				neighborParticle.velocity.z = -invmass * deltax.z;

			}
		}
	}

	/*printf("FRICTION: %f %f %f.\n", deltax.x, deltax.y, deltax.z);*/
	return deltax;
}

void movewallz(std::vector<ParticleStruct> &p_list) {
	int num_particles = p_list.size();

	for (int i = 0; i < num_particles; i++) {
		if (p_list[i].pencil) {
			p_list[i].current_position.z = p_list[i].current_position.z + positions.z;
		}
	}
}

//Ported to opencl, needs revision
void movewallx(std::vector<ParticleStruct> &p_list) {
	int num_particles = p_list.size();

	for (int i = 0; i < num_particles; i++) {
		if (p_list[i].pencil) {
			glm::vec3 temp(glm::vec3(p_list[i].varx, p_list[i].vary, -1.0f));
			p_list[i].current_position.x = (positions.x + temp.x);
			p_list[i].current_position.y = (positions.y + temp.y);
			p_list[i].current_position.z = (positions.z + temp.z);
		}
	}
}

void movewally(std::vector<ParticleStruct> &p_list) {
	int num_particles = p_list.size();

	for (int i = 0; i < num_particles; i++) {
		if (p_list[i].pencil) {
			p_list[i].current_position.y = p_list[i].current_position.y + move_wally;
		}
	}
}

//Aux
std::vector<int> getNeighbourBins(int initialBin, cl_int* numBins, int totalBins) {

	//Neighbour List
	std::vector<int> neighbourBins;

	//Base for each Z
	int baseBin;
	int currentBin;

	for (int z = -1; z <= 1; z++) {

		//Defines base for this Z
		baseBin = initialBin + (z * (numBins[0] * numBins[1]));

		//Central Bin
		currentBin = baseBin;
		if (currentBin >= 0 && currentBin < totalBins) neighbourBins.push_back(currentBin);

		//Right Bin
		currentBin = baseBin + 1;
		if (currentBin >= 0 && currentBin < totalBins) neighbourBins.push_back(currentBin);

		//Left Bin
		currentBin = baseBin - 1;
		if (currentBin >= 0 && currentBin < totalBins) neighbourBins.push_back(currentBin);

		//Up Bin
		currentBin = baseBin - numBins[0];
		if (currentBin >= 0 && currentBin < totalBins) neighbourBins.push_back(currentBin);

		//Down Bin
		currentBin = baseBin + numBins[0];
		if (currentBin >= 0 && currentBin < totalBins) neighbourBins.push_back(currentBin);

	}

	return neighbourBins;
}
