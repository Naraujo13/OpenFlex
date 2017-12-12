typedef struct ParticleStruct {
	glm::vec3 startPosition;
	glm::vec3 predicted_position;
	glm::vec3 velocity;
	glm::vec3 delta_p;
	float mass;
	float lambda;
	float rho;
	float C;
	float phase;
	bool teardrop;
	bool isRigidBody;
	bool pencil;
	bool isCollidingWithRigidBody;
};