// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <pbf.hpp>
#include <CL\cl.hpp>

#define PI 3.1415f
#define EPSILON 600.0f
#define ITER 2
//#define REST 6378.0f
#define DT 0.0083f
#define PARTICLE_COUNT_X 10
#define PARTICLE_COUNT_Y 2
#define PARTICLE_COUNT_Z 10


// Include GLEW
#include <GL/glew.h>
#include <unordered_map>
#include <math.h>

// Include GLFW
#include <glfw3.h>
GLFWwindow* g_pWindow;
unsigned int g_nWidth = 1024, g_nHeight = 768;

// Include AntTweakBar
#include <AntTweakBar.h>
TwBar *g_pToolBar;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtx/perpendicular.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;
#include <shader.hpp>
#include <texture.hpp>
#include <controls.hpp>
#include <objloader.hpp>
#include <vboindexer.hpp>
#include <glerror.hpp>

void WindowSizeCallBack(GLFWwindow *pWindow, int nWidth, int nHeight) {

	g_nWidth = nWidth;
	g_nHeight = nHeight;
	glViewport(0, 0, g_nWidth, g_nHeight);
	TwWindowSize(g_nWidth, g_nHeight);
}

//struct Particle
//{
//	glm::vec2 pos; 					// position
//	glm::vec2 vel; 					// velocity
//	float m; 						// mass
//	glm::vec2 dp;					// Delta p, to be used during the particle projection
//	float lambda;					// particle lambda
//	float rho;						// density at the particle
//	float C;						// density constraint for the particle
//	std::vector< unsigned int > n;	// neighbor particles' indexes
//	float hash;						// the hash for the particle
//};

typedef std::unordered_multimap< int, int > Hash;
extern Hash hash_table;
extern SpatialHash spatial_hash;

extern std::vector<Particle> particlesList;
extern std::vector< Particle > predict_p;
//std::vector<float> g_grid;
extern float g_xmax;
extern float g_xmin;
extern float g_ymax;
extern float g_ymin;
extern float g_zmax;
extern float g_zmin;
extern float particle_size;
extern float g_h;
extern float POW_H_9; // h^9
extern float POW_H_6; // h^6
					  //float rigidBody = 10;
					  //time_t g_initTime = time(0);
extern float rotatey;
extern float rotatex;
extern float rotatez;
extern float g_k;
extern float g_dq;
extern float gota;
extern float viscosityC;
extern float vorticityEps;
extern float resetParticles;
extern glm::vec3 gravity;
extern float boundary;
extern float masswall;

extern float wallscalex;
extern float wallscaley;
extern float wallscalez;
extern float wallLightx;
extern float wallLighty;
extern float wallLightz;
extern int GRID_RESOLUTION;
extern float adhesioncoeff;
extern float cohesioncoeff;
extern float move_wallz;
extern float move_wallx;
extern float move_wally;
extern float h_adhesion;
extern float gravity_y;
extern float kinetic;
extern float stattc;
extern float REST;
extern float wall_h;

extern glm::vec3 positions;
extern glm::vec3 direction;

void Algorithm() {

	double timeb4 = glfwGetTime();

	gravity.y = gravity_y;

	if (gota == 1) {
		hose();
		gota = 2;
	}

	if (resetParticles == 1) {
		particlesList.clear();
		predict_p.clear();
		cube();
	}

	int npart = particlesList.size();

	//Apply forces to non-rigidBody and not colliding with rigidBody
	#pragma omp parallel for
	for (int i = 0; i < npart; i++) {
		if (!predict_p[i].isRigidBody && !predict_p[i].isCollidingWithRigidBody) {
			predict_p[i].velocity = particlesList[i].velocity + DT * gravity;
			predict_p[i].current_position = particlesList[i].current_position + DT * predict_p[i].velocity;
		}
		/*else
		predict_p[i].teardrop = false;*/
	}

	timeb4 = glfwGetTime();
	//newBuildHashTable(predict_p, spatial_hash);
	BuildHashTable(predict_p, hash_table);
	//std::cout << "Time on building hash table: " << glfwGetTime() - timeb4 << " seconds" << std::endl;


	timeb4 = glfwGetTime();
	//newSetUpNeighborsLists(predict_p, spatial_hash);
	SetUpNeighborsLists(predict_p, hash_table);
	//std::cout << "Time on setting neighbours " << glfwGetTime() - timeb4 << " seconds" << std::endl << std::endl;

	//for (int i = 0; i < npart; i++) {
		//std::cout << "Particula " << i << " -> " << predict_p[i].allNeighbours.size() << " vizinhos\n";
	//	getchar();
	//}
	
	

	int iter = 0;

	//Solver Iterations
	while (iter < ITER) {	
		//std::cout << "Iter " << iter << std::endl;
		//For all particles -> density estimations
		#pragma omp parallel for
		for (int i = 0; i < npart; i++) {

			//If particle isnt rigidBody or colliding with rigidBody
			if (!predict_p[i].isRigidBody && !predict_p[i].isCollidingWithRigidBody) {

				//Estimate density of current particle
				DensityEstimator(predict_p, i);

				//Sets density constraint
				predict_p[i].C = predict_p[i].rho / REST - 1;

				//?
				float sumNabla = NablaCSquaredSumFunction(predict_p[i], predict_p);
				predict_p[i].lambda = -predict_p[i].C / (sumNabla + EPSILON);

			}

		}

		//std::cout << "before dp\n";

		//Calculate delta P for prediction
		CalculateDp(predict_p);


		//std::cout << "before collision\n";

		//Collision Detection and Response
		CollisionDetectionResponse(predict_p);
		
		//For each particle
		#pragma omp parallel for
		for (int i = 0; i < npart; i++) {
			//If particle isnt rigidBody or colliding with rigidBody
			if (!predict_p[i].isRigidBody && !predict_p[i].isCollidingWithRigidBody)
				//Predict new particle position
				predict_p[i].current_position = predict_p[i].current_position + predict_p[i].delta_p;
		}

		iter++;
	}


	//For each particle
	#pragma omp parallel for
	for (int i = 0; i < npart; i++) {
		
		//If particle isnt rigidBody or colliding with rigidBody
		if (!predict_p[i].isRigidBody && !predict_p[i].isCollidingWithRigidBody) {

			//Gets velocity based on original and predicted position
			predict_p[i].velocity = (1 / DT) * (predict_p[i].current_position - particlesList[i].current_position);
			
			//If it has neighbours
			if (predict_p[i].rigidBodyNeighbours.size() > 0) {
				
				//Applies adhesion and friction factors to velocity
				predict_p[i].velocity += adhesion(predict_p[i], predict_p);
				predict_p[i].velocity += particleFriction(predict_p[i], predict_p, i);
			}

			//Applies surface tension, vorticity and viscosity to velocity
			predict_p[i].velocity += surfaceTension(predict_p[i], predict_p) * DT;
			predict_p[i].velocity += VorticityConfinement(predict_p[i], predict_p) * DT;
			predict_p[i].velocity += XSPHViscosity(predict_p[i], predict_p) * DT;
		}

		//Clear neighbours
		predict_p[i].allNeighbours.clear();
		predict_p[i].rigidBodyNeighbours.clear();
		predict_p[i].notRigidBodyNeighbours.clear();
	}


	movewallx(predict_p);
	movewally(predict_p);
	movewallz(predict_p);

	particlesList = predict_p;
}

std::string GetPlatformName(cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetPlatformInfo(id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data()), nullptr);

	return result;
}

std::string GetDeviceName(cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetDeviceInfo(id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data()), nullptr);

	return result;
}

void CheckError(cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit(1);
	}
}

const char *kernelChar = "\n" \
"__kernel void duplica(__global float* in, __global float* out)\n" \
"{\n" \
"int temp = in[0];\n" \
"out[0] = temp + temp\n;" \
"}\n";

int main(void)
{
	//----------- OpenCL -------------

	//Get Number of Platforms
	cl_uint platformIdCount = 0;
	clGetPlatformIDs(0, nullptr, &platformIdCount);
	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		return 1;
	}
	else
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	
	//Get Platforms
	std::vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs (platformIdCount, platformIds.data(), nullptr);
	for (cl_uint i = 0; i < platformIdCount; ++i)
		std::cout << "\t (" << (i + 1) << ") : " << GetPlatformName(platformIds[i]) << std::endl;
	

	//Get Number of Devices
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);
	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		return 1;
	}
	else 
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	
	//Get Devices
	std::vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);
	for (cl_uint i = 0; i < deviceIdCount; ++i)
		std::cout << "\t (" << (i + 1) << ") : " << GetDeviceName(deviceIds[i]) << std::endl;


	//Creates context properties based on first device
	const cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		reinterpret_cast<cl_context_properties> (platformIds[0]),
		0, 0
	};

	//Creates variable to error string
	cl_int errorCode = 0;

	cl_context context = clCreateContext(
		contextProperties, deviceIdCount,
		deviceIds.data(), nullptr,
		nullptr, &errorCode);
	CheckError(errorCode);

	//Alocates vector and buffer
	float *h_input = (float *) malloc(sizeof(float) * 2);
	float *h_output = (float *)malloc(sizeof(float) * 2);
	h_input[0] = 2;
	h_input[1] = 3;
	h_output[0] = 0;
	h_output[1] = 0;
	cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 2, h_input, &errorCode);
	CheckError(errorCode);
	cl_mem d_output = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 2, h_input, &errorCode);
	CheckError(errorCode);

	// Create a command commands
	cl_command_queue queue = clCreateCommandQueue(context, deviceIds[1], 0, &errorCode);
	CheckError(errorCode);

	//Creates Program
	cl_program program = clCreateProgramWithSource(
		context, 1, (const char **)& kernelChar, NULL, &errorCode
	);
	CheckError(errorCode);

	// Build the program executable
	errorCode = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CheckError(errorCode);

	//Creates Kernel 
	cl_kernel kernel = clCreateKernel(program, "duplica", &errorCode);
	CheckError(errorCode);

	//Set Kernel Args
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);

	//Enqueues for execution
	const size_t globalWorkSize[] = { 2, 0, 0 };
	CheckError(clEnqueueNDRangeKernel(
		queue, 
		kernel,
		1, 
		nullptr,
		globalWorkSize,
		nullptr,
		0, 
		nullptr,
		nullptr
	));

	std::cout << "In before: (1) " << h_input[0] << "\t(2) " << h_input[1] << std::endl;
	std::cout << "Out before: (1) " << h_output[0] << "\t(2) " << h_output[1] << std::endl;

	//Gets results back
	clEnqueueReadBuffer(
		queue,		//Command Queue
		d_output, //Device source
		CL_TRUE, //Blocking?
		0,		//Offset in bytes from start of array
		sizeof(float) * 2, //Buffer 
		h_output,	//Host target
		0,	//Num of events to be executed before this command -> 0 == doesnt wait
		NULL, //Event List to be executed before this command -> NULL == doesnt wait
		NULL //Object Event to be returned that identify this command that can be used to query event status or queue a wait

	);

	std::cout << "In after: (1) " << h_input[0] << "\t(2) " << h_input[1] << std::endl;
	std::cout << "Out after: (1) " << h_output[0] << "\t(2) " << h_output[1] << std::endl;

	int nUseMouse = 0;
	InitParticleList();
	cube();

	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	g_pWindow = glfwCreateWindow(g_nWidth, g_nHeight, "CG UFFS", NULL, NULL);
	if (g_pWindow == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(g_pWindow);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	check_gl_error();//OpenGL error from GLEW

					 // Initialize the GUI
	TwInit(TW_OPENGL_CORE, NULL);
	TwWindowSize(g_nWidth, g_nHeight);

	// Set GLFW event callbacks. I removed glfwSetWindowSizeCallback for conciseness
	glfwSetMouseButtonCallback(g_pWindow, (GLFWmousebuttonfun)TwEventMouseButtonGLFW); // - Directly redirect GLFW mouse button events to AntTweakBar
	glfwSetCursorPosCallback(g_pWindow, (GLFWcursorposfun)TwEventMousePosGLFW);          // - Directly redirect GLFW mouse position events to AntTweakBar
	glfwSetScrollCallback(g_pWindow, (GLFWscrollfun)TwEventMouseWheelGLFW);    // - Directly redirect GLFW mouse wheel events to AntTweakBar
	glfwSetKeyCallback(g_pWindow, (GLFWkeyfun)TwEventKeyGLFW);                         // - Directly redirect GLFW key events to AntTweakBar
	glfwSetCharCallback(g_pWindow, (GLFWcharfun)TwEventCharGLFW);                      // - Directly redirect GLFW char events to AntTweakBar
	glfwSetWindowSizeCallback(g_pWindow, WindowSizeCallBack);

	//create the toolbar
	g_pToolBar = TwNewBar("CG UFFS ToolBar");
	// Add 'speed' to 'bar': it is a modifable (RW) variable of type TW_TYPE_DOUBLE. Its key shortcuts are [s] and [S].
	double speed = 0.0;
	TwAddVarRW(g_pToolBar, "speed", TW_TYPE_DOUBLE, &speed, " label='Rot speed' min=0 max=2 step=0.01 keyIncr=s keyDecr=S help='Rotation speed (turns/second)' ");
	// Add 'bgColor' to 'bar': it is a modifable variable of type TW_TYPE_COLOR3F (3 floats color)
	vec3 oColor(0.0f);
	TwAddVarRW(g_pToolBar, "bgColor", TW_TYPE_COLOR3F, &oColor[0], " label='Background color' ");
	TwAddVarRW(g_pToolBar, "g_h", TW_TYPE_FLOAT, &g_h, " label='H radius' min=0.1 max=5 step=0.01 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");
	/*TwAddVarRW(g_pToolBar, "rotatey", TW_TYPE_FLOAT, &rotatey, " label='rotation y of rigidBody' min=-360 max=360 step=1.0 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "rotatex", TW_TYPE_FLOAT, &rotatex, " label='rotation x of rigidBody' min=-360 max=360 step=1.0 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "rotatez", TW_TYPE_FLOAT, &rotatez, " label='rotation y of rigidBody' min=-360 max=360 step=1.0 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");*/
	TwAddVarRW(g_pToolBar, "g_zmax", TW_TYPE_FLOAT, &g_zmax, " label='position z of wall' min=-13 max=13 step=0.05 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "g_xmax", TW_TYPE_FLOAT, &g_xmax, " label='position x of wall' min=-40 max=40 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	/*TwAddVarRW(g_pToolBar, "g_k", TW_TYPE_FLOAT, &g_k, " label='k for scorr' min=-13 max=13 step=0.0001 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "g_dq", TW_TYPE_FLOAT, &g_dq, " label='dq for scorr' min=-13 max=13 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");*/
	TwAddVarRW(g_pToolBar, "particle_size", TW_TYPE_FLOAT, &particle_size, " label='particle_size' min=0 max=5 step=0.01 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "viscosityC", TW_TYPE_FLOAT, &viscosityC, " label='viscosityC' min=0 max=1 step=0.0001 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "vorticityEps", TW_TYPE_FLOAT, &vorticityEps, " label='vorticityEps' min=-13 max=13 step=0.00001 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "gota", TW_TYPE_FLOAT, &gota, " label='gota' min=0 max=2 step=1 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "resetParticles", TW_TYPE_FLOAT, &resetParticles, " label='resetParticles' min=0 max=2 step=1 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "wall z", TW_TYPE_FLOAT, &move_wallz, " label='particle z' min=-0.1 max=0.1 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "wall x", TW_TYPE_FLOAT, &move_wallx, " label='particle x' min=-0.1 max=0.1 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "wall y", TW_TYPE_FLOAT, &move_wally, " label='particle y' min=-0.1 max=0.1 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	/*TwAddVarRW(g_pToolBar, "wallscalex", TW_TYPE_FLOAT, &wallscalex, " label='wallscalex' min=0 max=5 step=0.01 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "wallscaley", TW_TYPE_FLOAT, &wallscaley, " label='wallscaley' min=0 max=5 step=0.01 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "wallscalez", TW_TYPE_FLOAT, &wallscalez, " label='wallscalez' min=0 max=5 step=0.01 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "wallLightx", TW_TYPE_FLOAT, &wallLightx, " label='wallLightx' min=0 max=5 step=0.01 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "wallLighty", TW_TYPE_FLOAT, &wallLighty, " label='wallLighty' min=0 max=5 step=0.01 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "wallLightz", TW_TYPE_FLOAT, &wallLightz, " label='wallLightz' min=0 max=5 step=0.01 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");*/
	//TwAddVarRW(g_pToolBar, "GRID_RESOLUTION", TW_TYPE_INT32, &GRID_RESOLUTION, " label='GRID_RESOLUTION' min=0 max=120 step=1.0 keyIncr=h keyDecr=H help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "adhesioncoeff", TW_TYPE_FLOAT, &adhesioncoeff, " label='adhesioncoeff' min=0 max=50 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "cohesioncoeff", TW_TYPE_FLOAT, &cohesioncoeff, " label='cohesioncoeff' min=0 max=50 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "gravity_y", TW_TYPE_FLOAT, &gravity_y, " label='gravity' min=-10 max=50 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "h_adhesion", TW_TYPE_FLOAT, &h_adhesion, " label='h_adhesion' min=0 max=5.5 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "kinetic", TW_TYPE_FLOAT, &kinetic, " label='kinetic' min=0 max=105 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "stattc", TW_TYPE_FLOAT, &stattc, " label='stattc' min=0 max=105 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "REST", TW_TYPE_FLOAT, &REST, " label='REST' min=0 max=7000 step=1.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "masswall", TW_TYPE_FLOAT, &masswall, " label='masswall' min=0 max=50 step=0.01 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");
	TwAddVarRW(g_pToolBar, "wall_h", TW_TYPE_FLOAT, &wall_h, " label='wall_h' min=0 max=1 step=0.001 keyIncr=r keyDecr=R help='Rotation speed (turns/second)' ");




	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(g_pWindow, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetCursorPos(g_pWindow, g_nWidth / 2, g_nHeight / 2);

	// Dark blue background
	glClearColor(0.7f, 0.9f, 0.9f, 0.4f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// Create and compile our GLSL program from the shaders
	GLuint standardProgramID = LoadShaders("shaders/StandardShading.vertexshader", "shaders/StandardShading.fragmentshader");
	GLuint wallProgramID = LoadShaders("shaders/wallShading.vertexshader", "shaders/wallShading.fragmentshader");
	/*BuildGrid();*/

	// Load the texture
	//GLuint Texture = loadDDS("mesh/uvmap.DDS");

	// Get a handle for our "myTextureSampler" uniform
	//GLuint TextureID = glGetUniformLocation(standardProgramID, "myTextureSampler");

	// Read our .obj file
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals;
	bool res = loadOBJ("mesh/esfera.obj", vertices, uvs, normals);

	std::vector<unsigned short> indices;
	std::vector<glm::vec3> indexed_vertices;
	std::vector<glm::vec2> indexed_uvs;
	std::vector<glm::vec3> indexed_normals;
	indexVBO(vertices, uvs, normals, indices, indexed_vertices, indexed_uvs, indexed_normals);

	// Load it into a VBO

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, indexed_vertices.size() * sizeof(glm::vec3), &indexed_vertices[0], GL_STATIC_DRAW);

	GLuint normalbuffer;
	glGenBuffers(1, &normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, indexed_normals.size() * sizeof(glm::vec3), &indexed_normals[0], GL_STATIC_DRAW);

	// Generate a buffer for the indices as well
	GLuint elementbuffer;
	glGenBuffers(1, &elementbuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);

	//paredes

	std::vector<glm::vec3> wall_vertices;
	std::vector<glm::vec2> wall_uvs;
	std::vector<glm::vec3> wall_normals;
	bool wallres = loadOBJ("mesh/cube2.obj", wall_vertices, wall_uvs, wall_normals);

	std::vector<unsigned short> wall_indices;
	std::vector<glm::vec3> wall_indexed_vertices;
	std::vector<glm::vec2> wall_indexed_uvs;
	std::vector<glm::vec3> wall_indexed_normals;
	indexVBO(wall_vertices, wall_uvs, wall_normals, wall_indices, wall_indexed_vertices, wall_indexed_uvs, wall_indexed_normals);

	// Load it into a VBO

	GLuint wallbuffer;
	glGenBuffers(1, &wallbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, wallbuffer);
	glBufferData(GL_ARRAY_BUFFER, wall_indexed_vertices.size() * sizeof(glm::vec3), &wall_indexed_vertices[0], GL_STATIC_DRAW);

	GLuint wallnormalbuffer;
	glGenBuffers(1, &wallnormalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, wallnormalbuffer);
	glBufferData(GL_ARRAY_BUFFER, wall_indexed_normals.size() * sizeof(glm::vec3), &wall_indexed_normals[0], GL_STATIC_DRAW);

	GLuint wallelementbuffer;
	glGenBuffers(1, &wallelementbuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wallelementbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, wall_indices.size() * sizeof(unsigned short), &wall_indices[0], GL_STATIC_DRAW);

	/*GLuint uvbuffer;
	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, indexed_uvs.size() * sizeof(glm::vec2), &indexed_uvs[0], GL_STATIC_DRAW);*/

	// Get a handle for our "LightPosition" uniform
	glUseProgram(standardProgramID);

	GLuint LightID = glGetUniformLocation(standardProgramID, "LightPosition_worldspace");

	// For speed computation
	double lastTime = glfwGetTime();
	int nbFrames = 0;

	do {

		check_gl_error();

		/* -- Keyboard Controls -- */

		//Lock/Unlock Mouse (Right Mouse Button)
		if (glfwGetMouseButton(g_pWindow, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS)
			nUseMouse = 0;
		else
			nUseMouse = 1;
		
		//Reset Particles (Mouse wheel?)
		if (glfwGetMouseButton(g_pWindow, GLFW_MOUSE_BUTTON_MIDDLE) != GLFW_PRESS)
			resetParticles = 0;
		else
			resetParticles = 1;

		//??????? (R)
		if (glfwGetKey(g_pWindow, GLFW_KEY_R) == GLFW_PRESS)
			gota = 1;
		else
			gota = 0;
		
		/* ----------------------- */

		
		/*if (glfwGetKey(g_pWindow, GLFW_KEY_R) != GLFW_PRESS)
		render = false;
		else
		render = true;*/

		/*if (render) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}
		else
		glDisable(GL_BLEND);*/
		
		
	
		//FPS measure
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lastTime >= 1.0) { // If last prinf() was more than 1sec ago
											 // printf and reset
			printf("%f ms/frame\n", 1000.0 / double(nbFrames));
			nbFrames = 0;
			lastTime += 1.0;
		}


		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Compute the MVP matrix from keyboard and mouse input
		positions = computeMatricesFromInputs(nUseMouse, g_nWidth, g_nHeight);
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();

		/* -- Shader -- */

		GLuint MatrixID = glGetUniformLocation(standardProgramID, "MVP");
		glm::mat4 MVP = ProjectionMatrix * ViewMatrix;
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		//Standard error check
		check_gl_error();

		glUseProgram(standardProgramID);
		MatrixID = glGetUniformLocation(standardProgramID, "MVP");
		GLuint ViewMatrixID = glGetUniformLocation(standardProgramID, "V");
		GLuint ModelMatrixID = glGetUniformLocation(standardProgramID, "M");

		//Light
		glm::vec3 lightPos = glm::vec3(0, 100, 0);
		glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

		/* ------------ */

		// Bind our texture in Texture Unit 0
		//glActiveTexture(GL_TEXTURE0);
		//glBindTexture(GL_TEXTURE_2D, Texture);
		//// Set our "myTextureSampler" sampler to user Texture Unit 0
		//glUniform1i(TextureID, 0);


		/* -- OpenGL Calls -- */

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : UVs
		//glEnableVertexAttribArray(1);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		//glVertexAttribPointer(
		//	1,                                // attribute
		//	2,                                // size
		//	GL_FLOAT,                         // type
		//	GL_FALSE,                         // normalized?
		//	0,                                // stride
		//	(void*)0                          // array buffer offset
		//	);

		// 3rd attribute buffer : normals
		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
		glVertexAttribPointer(
			2,                                // attribute
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		// Index buffer
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);

		glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &ViewMatrix[0][0]);

		/* -------------------------------- */


		glm::mat4 ModelMatrix = glm::mat4(1.0f); //Usar posi��o aleat�ria
		ModelMatrix[0][0] = particle_size; //Escala do modelo (x)
		ModelMatrix[1][1] = particle_size; //Escala do modelo (y)
		ModelMatrix[2][2] = particle_size; //Escala do modelo (z)
		
		//double timeb4 = glfwGetTime();
		Algorithm();
		//std::cout << "Time on algorithm: " << glfwGetTime() - timeb4 << " seconds" << std::endl;

		//Render
		GLuint particleColor = glGetUniformLocation(standardProgramID, "particleColor");
		glUniform3f(particleColor, 0.0f, 0.5f, 0.9f);
		
		//timeb4 = glfwGetTime();

		/* -- Draw particles -- */
		for (int index = 0; index < particlesList.size(); index++) {			

			//for
			ModelMatrix[3][0] = particlesList[index].current_position.x; //posi��o x
			ModelMatrix[3][1] = particlesList[index].current_position.y; //posi��o y
			ModelMatrix[3][2] = particlesList[index].current_position.z; //posi��o z

			if (particlesList[index].teardrop)
				glUniform3f(particleColor, 1.0f, 0.0f, 0.0f);

			glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

			// Send our transformation to the currently bound shader,
			// in the "MVP" uniform
			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
			glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);


			// Draw the triangles !
			/*if (!particles[index].rigidBody){*/
			glDrawElements(
				GL_TRIANGLES,        // mode
				indices.size(),      // count
				GL_UNSIGNED_SHORT,   // type
				(void*)0             // element array buffer offset
			);
			//}
			//endfor
		}
		//std::cout << "Time particles loop: " << glfwGetTime() - timeb4 << " seconds" << std::endl;


		/* ------ Room ------- */

		glUseProgram(wallProgramID);

		GLuint wallLightID = glGetUniformLocation(wallProgramID, "LightPosition_worldspace");
		MatrixID = glGetUniformLocation(wallProgramID, "MVP");
		ViewMatrixID = glGetUniformLocation(wallProgramID, "V");
		ModelMatrixID = glGetUniformLocation(wallProgramID, "M");

		glm::vec3 wallLightPos = glm::vec3(1.90f, 2.0f, 3.5f);
		glUniform3f(wallLightID, wallLightPos.x, wallLightPos.y, wallLightPos.z);

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, wallbuffer);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// 2nd attribute buffer : UVs
		//glEnableVertexAttribArray(1);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		//glVertexAttribPointer(
		//	1,                                // attribute
		//	2,                                // size
		//	GL_FLOAT,                         // type
		//	GL_FALSE,                         // normalized?
		//	0,                                // stride
		//	(void*)0                          // array buffer offset
		//	);

		// 3rd attribute buffer : normals
		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, wallnormalbuffer);
		glVertexAttribPointer(
			2,                                // attribute
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		// Index buffer
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wallelementbuffer);
		glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &ViewMatrix[0][0]);


		//Prepara modelmatriz que ser� usada para as paredes da sala
		ModelMatrix = glm::mat4(1.0f); //Usar posi��o aleat�ria
		ModelMatrix[0][0] = wallscalex * (g_ymax / 2); //Escala do modelo (x)
		ModelMatrix[1][1] = wallscaley * (g_ymax / 2); //Escala do modelo (y)
		ModelMatrix[2][2] = wallscalez * (g_ymax / 2); //Escala do modelo (z)

		
		/* -- Draw Back Wall -- */

		ModelMatrix[3][0] = g_xmin; //posi��o x
		ModelMatrix[3][1] = g_ymin;//posi��o y
		ModelMatrix[3][2] = g_zmin;//posi��o z

		ModelMatrix = glm::rotate(ModelMatrix, rotatex, glm::vec3(1, 0, 0));
		ModelMatrix = glm::rotate(ModelMatrix, rotatey, glm::vec3(0, 1, 0));
		ModelMatrix = glm::rotate(ModelMatrix, rotatez, glm::vec3(0, 0, 1));

		MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader,
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
		glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);


		// Draw the triangles !
		glDrawElements(
			GL_TRIANGLES,				// mode
			wall_indices.size(),		// count
			GL_UNSIGNED_SHORT,			// type
			(void*)0					// element array buffer offset
		);

		/* ----------------- */




		/* -- Draw Left Wall -- */

		ModelMatrix[3][0] = g_xmin; //posi��o x
		ModelMatrix[3][1] = g_ymin;//posi��o y
		ModelMatrix[3][2] = g_zmin;//posi��o z

		ModelMatrix = glm::rotate(ModelMatrix, 90.0f, glm::vec3(1, 0, 0));
		ModelMatrix = glm::rotate(ModelMatrix, 0.0f, glm::vec3(0, 1, 0));
		ModelMatrix = glm::rotate(ModelMatrix, rotatez, glm::vec3(0, 0, 1));

		MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader,
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
		glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);


		// Draw the triangles !
		glDrawElements(
			GL_TRIANGLES,				// mode
			wall_indices.size(),		// count
			GL_UNSIGNED_SHORT,			// type
			(void*)0					// element array buffer offset
		);

		
		/* ----------------- */



		/* -- Draw Left Wall -- */

		ModelMatrix[3][0] = g_xmin; //posi��o x
		ModelMatrix[3][1] = g_ymin;//posi��o y
		ModelMatrix[3][2] = g_zmin;//posi��o z

		ModelMatrix = glm::rotate(ModelMatrix, -90.0f, glm::vec3(1, 0, 0));
		ModelMatrix = glm::rotate(ModelMatrix, -90.0f, glm::vec3(0, 1, 0));
		ModelMatrix = glm::rotate(ModelMatrix, rotatez, glm::vec3(0, 0, 1));

		MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader,
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
		glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);


		// Draw the triangles !
		glDrawElements(
			GL_TRIANGLES,				// mode
			wall_indices.size(),		// count
			GL_UNSIGNED_SHORT,			// type
			(void*)0					// element array buffer offset
		);

		/* ----------------- */



		/* -- Draw Right Wall -- */

		ModelMatrix[3][0] = g_xmax; //posi��o x
		ModelMatrix[3][1] = g_ymin;//posi��o y
		ModelMatrix[3][2] = g_zmin;//posi��o z

		ModelMatrix = glm::rotate(ModelMatrix, rotatex, glm::vec3(1, 0, 0));
		ModelMatrix = glm::rotate(ModelMatrix, 0.0f, glm::vec3(0, 1, 0));
		ModelMatrix = glm::rotate(ModelMatrix, rotatez, glm::vec3(0, 0, 1));

		MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader,
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
		glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);


		// Draw the triangles !
		glDrawElements(
			GL_TRIANGLES,				// mode
			wall_indices.size(),		// count
			GL_UNSIGNED_SHORT,			// type
			(void*)0					// element array buffer offset
		);

		/* ----------------- */




		glDisableVertexAttribArray(0);
		// glDisableVertexAttribArray(1); Canal das texturas
		glDisableVertexAttribArray(2);

		// Draw tweak bars
		TwDraw();

		// Swap buffers
		glfwSwapBuffers(g_pWindow);
		glfwPollEvents();

		/*double currentTimeNeigh = glfwGetTime();
		printf("%f tempo depois de entrar no for \n", double(currentTimeNeigh));*/


	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(g_pWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(g_pWindow) == 0);


	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	//glDeleteBuffers(1, &uvbuffer);
	glDeleteBuffers(1, &normalbuffer);
	glDeleteBuffers(1, &elementbuffer);
	glDeleteProgram(standardProgramID);
	//glDeleteTextures(1, &Texture);
	glDeleteVertexArrays(1, &VertexArrayID);

	// Terminate AntTweakBar and GLFW
	TwTerminate();
	glfwTerminate();

	return 0;
}


