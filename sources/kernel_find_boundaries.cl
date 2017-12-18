__kernel void findBoundaries(__global int *hash, __global int* numKeys, __global int *binBoundaries, __global int* numBins){

	int index = get_local_id(0);

    int i;

    if (index >= *numKeys)
        return;

    int previousBin;
    if (index > 0)
        previousBin = hash[index - 1];
    else
        previousBin = -1;
    int currentBin = hash[index];

    if (previousBin != currentBin){
        for (i = currentBin; i > previousBin && i >= 0; i--){
            binBoundaries[i] = index;   
        }
    }
}