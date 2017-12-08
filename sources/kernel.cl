__kernel void duplica (__global float* vetor)
{
	int temp = vetor[get_global_id];

   vetor[get_global_id] = temp + temp;

}