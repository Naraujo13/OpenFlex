#include "particle.hpp"

/* -- Particle Functions -- */

//Construtor
ParticleClass::ParticleClass() {}

ParticleClass::ParticleClass(
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

/* Compares two lists of ParticleStruct */
bool compareParticleStructLists(std::vector<ParticleStruct> l1, std::vector<ParticleStruct> l2) {

	struct ParticleStructCompare : public std::unary_function<ParticleStruct, bool>
	{
		explicit ParticleStructCompare(const ParticleStruct &baseline) : baseline(baseline) {}

		bool operator() (const ParticleStruct &arg)
		{
			return (baseline.current_position.x == arg.current_position.x &&
				baseline.current_position.y == arg.current_position.y &&
				baseline.current_position.z == arg.current_position.z &&
				baseline.hash == arg.hash);
		}
		ParticleStruct baseline;
	};


	std::cout << "Comparing struct lists..." << std::endl;
	if (l1.size() != l2.size()) {
		std::cerr << "\tERROR - List size different" << std::endl;
		return false;
	}

	for (int i = 0; i < l1.size(); i++) {

		ParticleStruct p1 = l1.at(i);
		ParticleStruct p2 = l2.at(i);


		if (std::find_if(l2.begin(), l2.end(), ParticleStructCompare(p1)) == l2.end()) {

			std::cerr << "\tERROR - Particle at position " << i << " of list 1 is not present in list 2" << std::endl;

			std::cerr << "\t\tP1("
				<< l1.at(i).current_position.x << ", "
				<< l1.at(i).current_position.y << ", "
				<< l1.at(i).current_position.z << ")" << std::endl;

			return false;

		}
		else if (std::find_if(l1.begin(), l1.end(), ParticleStructCompare(p2)) == l1.end()) {

			std::cerr << "\tERROR - Particle at position " << i << " of list 2 is not present in list 1" << std::endl;

			std::cerr << "\t\tP2("
				<< l2.at(i).current_position.x << ", "
				<< l2.at(i).current_position.y << ", "
				<< l2.at(i).current_position.z << ")" << std::endl;
			return false;

		}
	}

	return true;
}


/* Sequential QuickSort method to order particle vector based on their spatial has value */
//TODO: VERIFY NULLPOINTER ON VALUE AND PARTICLES; VERIFY MATCHING SIZE;
void quickSort(cl_int* value, ParticleStruct* particles, int start, int end) {
	int i, j, x, y;
	i = start;
	j = end;
	x = value[(start + end) / 2];

	while (i <= j) {
		while (value[i] < x && i < end) {
			i++;
		}
		while (value[j] > x && j > start) {
			j--;
		}
		if (i <= j) {
			y = value[i];
			ParticleStruct temp = particles[i];

			particles[i] = particles[j];
			value[i] = value[j];

			particles[j] = temp;
			value[j] = y;

			i++;
			j--;
		}
	}
	if (j > start) {
		quickSort(value, particles, start, j);
	}
	if (i < end) {
		quickSort(value, particles, i, end);
	}
}