#ifndef UTILS_H_
#define UTILS_H_

#include <filesystem>

#include <torch/torch.h>

#include "renderer.h"

// Initialization functions
void set_seed(int seed);
torch::Device get_device();
bool parse_arguments(int argc, char *argv[], std::filesystem::path &data_path,
                     std::filesystem::path &output_path);

// File handling functions
std::vector<char> load_binary_file(const std::filesystem::path &file_path);
torch::Tensor load_tensor(const std::filesystem::path &file_path);
float load_focal(const std::filesystem::path &file_path);
void save_image(const torch::Tensor &tensor,
                const std::filesystem::path &file_path);

// Rendering helper functions
void render_and_save_orbit_views(const NeRFRenderer &renderer, int N_frames,
                                 const std::filesystem::path &output_folder,
                                 float radius = 4.0f,
                                 float start_distance = 2.0f,
                                 float end_distance = 5.0f, int N_samples = 64);

// Transformation and pose functions
torch::Tensor create_spherical_pose(float azimuth, float elevation,
                                    float radius);
torch::Tensor create_translation_matrix(float t);
torch::Tensor create_phi_rotation_matrix(float phi);
torch::Tensor create_theta_rotation_matrix(float theta);

#endif // UTILS_H_
