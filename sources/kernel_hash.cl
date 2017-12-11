
__kernel void hashFunction(__global float* maxDim, __global float* numBins, __global float* binSize, __global float* pos, __global int* hash) {

	int index = get_local_id(0) * 3;

	float maxDimX = maxDim[0];
	float maxDimY = maxDim[1];
	float maxDimZ = maxDim[2];

	float x = pos[index];
	float y = pos[index + 1];
	float z = pos[index + 2];

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

	int hashValue = 0;
	int temp = numBins[0];

	hashValue += x;

	hashValue += (y * temp);

	temp *= numBins[1];
	temp *= z;
	hashValue += temp;

	hash[get_local_id(0)] = hashValue;

	printf("\tHash %d: %d\n", get_local_id(0), hashValue);

}