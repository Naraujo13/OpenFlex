#include <iostream>
#include "opencl_utils.hpp"

//Check Error Number and returns corresponding string error
void checkError(cl_int error)
{
	std::string errorString;
	switch (error) {
		// run-time and JIT compiler errors
		case 0: errorString = "CL_SUCCESS";
		case -1: errorString = "CL_DEVICE_NOT_FOUND";
		case -2: errorString = "CL_DEVICE_NOT_AVAILABLE";
		case -3: errorString = "CL_COMPILER_NOT_AVAILABLE";
		case -4: errorString = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: errorString = "CL_OUT_OF_RESOURCES";
		case -6: errorString = "CL_OUT_OF_HOST_MEMORY";
		case -7: errorString = "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: errorString = "CL_MEM_COPY_OVERLAP";
		case -9: errorString = "CL_IMAGE_FORMAT_MISMATCH";
		case -10: errorString = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: errorString = "CL_BUILD_PROGRAM_FAILURE";
		case -12: errorString = "CL_MAP_FAILURE";
		case -13: errorString = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: errorString = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: errorString = "CL_COMPILE_PROGRAM_FAILURE";
		case -16: errorString = "CL_LINKER_NOT_AVAILABLE";
		case -17: errorString = "CL_LINK_PROGRAM_FAILURE";
		case -18: errorString = "CL_DEVICE_PARTITION_FAILED";
		case -19: errorString = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
		case -30: errorString = "CL_INVALID_VALUE";
		case -31: errorString = "CL_INVALID_DEVICE_TYPE";
		case -32: errorString = "CL_INVALID_PLATFORM";
		case -33: errorString = "CL_INVALID_DEVICE";
		case -34: errorString = "CL_INVALID_CONTEXT";
		case -35: errorString = "CL_INVALID_QUEUE_PROPERTIES";
		case -36: errorString = "CL_INVALID_COMMAND_QUEUE";
		case -37: errorString = "CL_INVALID_HOST_PTR";
		case -38: errorString = "CL_INVALID_MEM_OBJECT";
		case -39: errorString = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: errorString = "CL_INVALID_IMAGE_SIZE";
		case -41: errorString = "CL_INVALID_SAMPLER";
		case -42: errorString = "CL_INVALID_BINARY";
		case -43: errorString = "CL_INVALID_BUILD_OPTIONS";
		case -44: errorString = "CL_INVALID_PROGRAM";
		case -45: errorString = "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: errorString = "CL_INVALID_KERNEL_NAME";
		case -47: errorString = "CL_INVALID_KERNEL_DEFINITION";
		case -48: errorString = "CL_INVALID_KERNEL";
		case -49: errorString = "CL_INVALID_ARG_INDEX";
		case -50: errorString = "CL_INVALID_ARG_VALUE";
		case -51: errorString = "CL_INVALID_ARG_SIZE";
		case -52: errorString = "CL_INVALID_KERNEL_ARGS";
		case -53: errorString = "CL_INVALID_WORK_DIMENSION";
		case -54: errorString = "CL_INVALID_WORK_GROUP_SIZE";
		case -55: errorString = "CL_INVALID_WORK_ITEM_SIZE";
		case -56: errorString = "CL_INVALID_GLOBAL_OFFSET";
		case -57: errorString = "CL_INVALID_EVENT_WAIT_LIST";
		case -58: errorString = "CL_INVALID_EVENT";
		case -59: errorString = "CL_INVALID_OPERATION";
		case -60: errorString = "CL_INVALID_GL_OBJECT";
		case -61: errorString = "CL_INVALID_BUFFER_SIZE";
		case -62: errorString = "CL_INVALID_MIP_LEVEL";
		case -63: errorString = "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: errorString = "CL_INVALID_PROPERTY";
		case -65: errorString = "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: errorString = "CL_INVALID_COMPILER_OPTIONS";
		case -67: errorString = "CL_INVALID_LINKER_OPTIONS";
		case -68: errorString = "CL_INVALID_DEVICE_PARTITION_COUNT";

			// extension errors
		case -1000: errorString = "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: errorString = "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: errorString = "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: errorString = "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: errorString = "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: errorString = "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: errorString = "Unknown OpenCL error";
	}
	if (error > 1)
		exit(0);
}

//Read Kernel From File
std::pair<const char*, ::size_t> readKernelFromFile(char* fileName, int errorCode) {
	FILE *fp;
	char *source_str;
	size_t source_size, program_size;

	fp = fopen(fileName, "rb");
	if (!fp) {
		printf("Failed to load hashKernel\n");

		std::pair<const char*, ::size_t> source(nullptr, 0);
		errorCode = 1;
		return source;
	}

	fseek(fp, 0, SEEK_END);
	program_size = ftell(fp);
	rewind(fp);
	source_str = (char*)malloc(program_size + 1);
	source_str[program_size] = '\0';
	fread(source_str, sizeof(char), program_size, fp);

	fseek(fp, 0L, SEEK_END);
	int size = ftell(fp);
	fclose(fp);

	errorCode = 0;

	//Print code
	printf("Source code (%d bytes):\n", size);
	for (int i = 0; i < program_size; i++) {
		printf("%c", source_str[i]);
	}
	printf("\n");

	std::pair<const char*, ::size_t> source(source_str, size);

	return source;
}

//Logs program build
void logProgramBuild(cl_program program, cl_device_id device_id) {
	 //------------------Getting log
	 printf("\n----------\nStarting Program Build log...\n");
	 // Determine the size of the log
	 size_t log_size;
	 clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	 // Allocate memory for the log
	 char *log = (char *)malloc(log_size);

	 // Get the log
	 clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

	 // Print the log
	 printf("%s\n\n--------\n\n", log);
	 //-----------------------
}


//Get Platform Name by Id
std::string getPlatformName(cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetPlatformInfo(id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data()), nullptr);

	return result;
}

//Get Device Name by Id
std::string getDeviceName(cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetDeviceInfo(id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data()), nullptr);

	return result;
}

//Print all platform info, verbose way
void printPlatformInfo(cl::Platform platform, cl_int platformId) {
	std::cout << "Platform ID: " << platformId << std::endl;
	std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
}

//Print all device info, verbose way
void printDeviceInfo(cl::Device device, cl_int deviceId) {
	std::cout << "\tDevice " << deviceId << ": " << std::endl;
	std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
	std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
	std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
	std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
	std::cout << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
	std::cout << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
	std::cout << "\t\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
	std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
	std::cout << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
}

