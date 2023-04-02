#include "model.h"
#include "renderer.h"
#include "utils.h"

constexpr int seed = 1;
constexpr int n_iters = 2000;
constexpr int plot_freq = 100;
constexpr int n_preview_frames = 5;
constexpr int n_final_frames = 35;

int main(int argc, char *argv[]) {
  // Parse command-line arguments
  std::filesystem::path data_path;
  std::filesystem::path output_path;
  if (!parse_arguments(argc, argv, data_path, output_path)) {
    return 1;
  }

  // Set the random seed
  set_seed(seed);

  // Determine device for computation
  torch::Device device = get_device();

  // Load data: images, poses, and focal length
  torch::Tensor images = load_tensor(data_path / "images.pt").to(device);
  torch::Tensor poses = load_tensor(data_path / "poses.pt").to(device);
  float focal = load_focal(data_path / "focal.pt");

  // Display information about the loaded data
  std::cout << "Images: " << images.sizes() << std::endl;
  std::cout << "Poses: " << poses.sizes() << std::endl;
  std::cout << "Focal length: " << focal << std::endl;

  // Create NeRF model and renderer
  NeRFModel model(device);
  NeRFRenderer renderer(model, images.size(1), images.size(2), focal, device);

  // Set up the optimizer
  torch::optim::Adam optimizer(model.parameters(),
                               torch::optim::AdamOptions(5e-4));

  // Train the NeRF model
  for (int i = 0; i < n_iters; i++) {
    // Sample a random image and its corresponding pose
    int img_i = std::rand() % images.size(0);
    auto target = images[img_i];
    auto pose = poses[img_i];

    // Perform forward pass and compute loss
    optimizer.zero_grad();
    auto rgb = renderer.render(pose, true);
    auto loss = torch::mse_loss(rgb, target);

    // Perform backward pass and update model parameters
    loss.backward();
    optimizer.step();

    // Log progress periodically
    if (i % plot_freq == 0) {
      torch::NoGradGuard no_grad;
      std::cout << "Iteration: " << i + 1 << " Loss: " << loss.item<float>()
                << std::endl;

      // Render and save orbiting preview for logging
      render_and_save_orbit_views(renderer, n_preview_frames, output_path,
                                  4.0f);
    }
  }

  std::cout << "Done" << std::endl;

  // Generate high-resolution rendering using the trained model
  torch::NoGradGuard no_grad;
  NeRFRenderer renderer_hd(model, 300, 300, focal, device);
  render_and_save_orbit_views(renderer_hd, n_final_frames, output_path, 2.1f,
                              0.8f, 3.2f);

  return 0;
}
