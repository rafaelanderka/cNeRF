#ifndef RENDERER_H_
#define RENDERER_H_

#include <torch/torch.h>

#include "model.h"

class NeRFRenderer {
public:
  NeRFRenderer(NeRFModel &model, int H, int W, float focal,
               const torch::Device &device);

  torch::Tensor render(const torch::Tensor &pose, bool randomize = false,
                       float start_distance = 2.0f, float end_distance = 5.0f,
                       int n_samples = 64, int batch_size = 64000) const;

private:
  typedef std::tuple<torch::Tensor, torch::Tensor> RayData;

  NeRFModel &model_;
  const torch::Device &device_;
  int H_;
  int W_;
  float focal_;

  RayData get_rays(const torch::Tensor &pose) const;
  torch::Tensor render_rays(const RayData &rays, bool randomize,
                            float start_distance, float end_distance,
                            int n_samples, int batch_size) const;
};

#endif // RENDERER_H_
