#ifndef MODEL_H_
#define MODEL_H_

#include <torch/torch.h>

class NeRFModel : public torch::nn::Module {
public:
  NeRFModel(const torch::Device &device = torch::kCPU, int L_embed = 6,
            int D = 8, int W = 256);

  torch::Tensor forward(const torch::Tensor &input);
  torch::Tensor add_positional_encoding(const torch::Tensor &x) const;

private:
  int L_embed_;
  torch::nn::Sequential model_;
  const torch::Device &device_;
};

#endif // MODEL_H_
