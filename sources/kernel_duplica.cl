__kernel void duplica(__global float* in, __global float* out) {
	int temp = in[0];
	out[0] = temp + temp;
}