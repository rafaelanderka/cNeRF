#include "utils.h"

#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Set up the random seed for reproducibility
void set_seed(int seed) {
  torch::manual_seed(seed);
  if (torch::cuda::is_available()) {
    torch::cuda::manual_seed(seed);
  }
}

// Determine the appropriate device for computation (CPU or GPU)
torch::Device get_device() {
  return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
}

bool parse_arguments(int argc, char *argv[], std::filesystem::path &data_path,
                     std::filesystem::path &output_path) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <data_path> <output_path>"
              << std::endl;
    return false;
  }

  data_path = argv[1];
  output_path = argv[2];
  return true;
}

std::vector<char> load_binary_file(const std::filesystem::path &file_path) {
  std::ifstream input(file_path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));
  input.close();
  return bytes;
}

torch::Tensor load_tensor(const std::filesystem::path &file_path) {
  std::vector<char> f = load_binary_file(file_path);
  torch::IValue x = torch::pickle_load(f);
  return x.toTensor();
}

float load_focal(const std::filesystem::path &file_path) {
  torch::Tensor focal_tensor = load_tensor(file_path);
  return focal_tensor.item<float>();
}

void save_image(const torch::Tensor &tensor,
                const std::filesystem::path &file_path) {
  // Assuming the input tensor is a 3-channel (HxWx3) image in the range [0, 1]
  auto height = tensor.size(0);
  auto width = tensor.size(1);
  auto max = tensor.max();
  auto min = tensor.min();
  // auto tensor_normalized = tensor.mul(255)
  auto tensor_normalized = ((tensor - min) / (max - min))
                               .mul(255)
                               .clamp(0, 255)
                               .to(torch::kU8)
                               .to(torch::kCPU)
                               .flatten()
                               .contiguous();
  cv::Mat image(cv::Size(width, height), CV_8UC3, tensor_normalized.data_ptr());
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  cv::imwrite(file_path.string(), image);
}

void render_and_save_orbit_views(const NeRFRenderer &renderer, int num_frames,
                                 const std::filesystem::path &output_folder,
                                 float radius, float start_distance,
                                 float end_distance, int n_samples) {
  float elevation = -30.0f;

  for (int i = 0; i < num_frames; i++) {
    float azimuth = static_cast<float>(i) * 360.0f / num_frames;
    auto pose = create_spherical_pose(azimuth, elevation, radius);

    auto rendered_image =
        renderer.render(pose, false, start_distance, end_distance, n_samples);

    std::string file_path =
        output_folder / ("frame_" + std::to_string(i) + ".png");
    save_image(rendered_image, file_path);
  }
}

torch::Tensor create_spherical_pose(float azimuth, float elevation,
                                    float radius) {
  float phi = elevation * (M_PI / 180.0f);
  float theta = azimuth * (M_PI / 180.0f);

  torch::Tensor c2w = create_translation_matrix(radius);
  c2w = create_phi_rotation_matrix(phi).matmul(c2w);
  c2w = create_theta_rotation_matrix(theta).matmul(c2w);
  c2w = torch::tensor({{-1.0f, 0.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 1.0f, 0.0f},
                       {0.0f, 1.0f, 0.0f, 0.0f},
                       {0.0f, 0.0f, 0.0f, 1.0f}})
            .matmul(c2w);

  return c2w;
}

torch::Tensor create_translation_matrix(float t) {
  torch::Tensor t_mat = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                                       {0.0f, 1.0f, 0.0f, 0.0f},
                                       {0.0f, 0.0f, 1.0f, t},
                                       {0.0f, 0.0f, 0.0f, 1.0f}});
  return t_mat;
}

torch::Tensor create_phi_rotation_matrix(float phi) {
  torch::Tensor phi_mat =
      torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                     {0.0f, std::cos(phi), -std::sin(phi), 0.0f},
                     {0.0f, std::sin(phi), std::cos(phi), 0.0f},
                     {0.0f, 0.0f, 0.0f, 1.0f}});
  return phi_mat;
}

torch::Tensor create_theta_rotation_matrix(float theta) {
  torch::Tensor theta_mat =
      torch::tensor({{std::cos(theta), 0.0f, -std::sin(theta), 0.0f},
                     {0.0f, 1.0f, 0.0f, 0.0f},
                     {std::sin(theta), 0.0f, std::cos(theta), 0.0f},
                     {0.0f, 0.0f, 0.0f, 1.0f}});
  return theta_mat;
}
