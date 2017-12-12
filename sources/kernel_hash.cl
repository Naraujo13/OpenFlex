typedef struct ParticleStruct {
	float3 current_position;
	float3 predicted_position;
	float3 velocity;
	float3 delta_p;
	float mass;
	float lambda;
	float rho;
	float C;
	float phase;
	float teardrop;
	float isRigidBody;
	float pencil;
	float isCollidingWithRigidBody;
};

__kernel void hashFunction(__global float* maxDim, __global float* numBins, __global float* binSize, __global struct ParticleStruct* pos, __global int* hash) {

	//printf("%.0f, %.0f, %.0f\n", maxDim[0], maxDim[1], maxDim[2]);
	
	int index = get_local_id(0);

	float maxDimX = maxDim[0];
	float maxDimY = maxDim[1];
	float maxDimZ = maxDim[2];

	float x = pos[index].current_position.x;
	float y = pos[index].current_position.y;
	float z = pos[index].current_position.z;

	if (x >= maxDimX)
		x = maxDimX - 1;
	else if (x <= -maxDimX)
		x = -maxDimX + 1;

	if (y >= maxDimY)
		y = maxDimY - 1;
	else if (y <= -maxDimY)
		y = -maxDimY + 1;

	if (z >= maxDimZ)
		z = maxDimZ - 1;
	else if (z <= -maxDimZ)
		z = -maxDimZ + 1;

	x = x + (maxDimX);
	y = -(y - (maxDimY));
	z = -(z - (maxDimZ));

	x = (int)(x / binSize[0]);
	y = (int)(y / binSize[1]);
	z = (int)(z / binSize[2]);
	printf("In Bin Coordinates: (%f, %f, %f)\n", x, y, z);

	int hashValue = 0;
	int temp = numBins[0];

	hashValue += x;

	hashValue += (y * temp);

	temp *= numBins[1];
	temp *= z;
	hashValue += temp;

	hash[index] = hashValue;

	printf("\tHash %d: %d\n", index, hashValue);

}