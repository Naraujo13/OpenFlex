#ifndef CONTROLS_HPP
#define CONTROLS_HPP

glm::vec3 computeMatricesFromInputs(int nUseMouse = 0, int nWidth = 1024, int nHeight = 768);
glm::mat4 getViewMatrix();
glm::mat4 getProjectionMatrix();

#endif