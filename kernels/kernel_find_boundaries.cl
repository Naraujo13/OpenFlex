__kernel void findBoundaries(__global int *hash, __global int* numKeys, __global int *binBoundaries, __global int* numBins){

	int index = get_local_id(0);
	int previousBin = -1;

	printf("Thread %d!\n", index);

    int i;

    if (index >= *numKeys){
		printf("\tIndex >= numKeys\n");
		return;
	}
	else{

		if (index > 0)
			previousBin = hash[index - 1];
	
		int currentBin = hash[index];

		printf("i %d\tp %d\tc %d\n", index, previousBin, currentBin);

		if (previousBin != currentBin){
			for (i = currentBin; i > previousBin && i >= 0; i--){
				binBoundaries[i] = index;
			}
		}

		//if (previousBin != currentBin){
		//	for (i = currentBin; i > previousBin && i >= 0; i--){
		//		binBoundaries[i] = index;   
		//	}
		//}

	}
}

		
