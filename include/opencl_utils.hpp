#pragma once
#ifndef OPENCL_UTILS_HPP
#define OPENCL_UTILS_HPP

//OpenCL include
#include <CL\cl.hpp>

//Error code helper
void checkError(cl_int error);

//Read Kernel from File
std::pair<const char*, ::size_t> readKernelFromFile(char* fileName, int errorCode);

//-- Helpers without C++ Wrapper --//
std::string getPlatformName(cl_platform_id id);
std::string getDeviceName(cl_device_id id);
void logProgramBuild(cl_program program, cl_device_id device_id);



#endif