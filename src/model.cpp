#include "model.h"

NeRFModel::NeRFModel(const torch::Device &device, int L_embed, int D, int W)
    : device_(device), L_embed_(L_embed) {
  // Create FFN
  auto input_dim = 3 + 3 * 2 * L_embed;
  model_->push_back(torch::nn::Linear(input_dim, W));
  model_->push_back(torch::nn::Functional(torch::relu));
  for (int i = 0; i < D - 2; i++) {
    model_->push_back(torch::nn::Linear(W, W));
    model_->push_back(torch::nn::Functional(torch::relu));
  }
  model_->push_back(torch::nn::Linear(W, 4));
  model_->to(device_);
  register_module("model", model_);

  this->to(device_);
}

torch::Tensor NeRFModel::forward(const torch::Tensor &input) {
  torch::Tensor x = model_->forward(input);
  return x;
}

torch::Tensor NeRFModel::add_positional_encoding(const torch::Tensor &x) const {
  std::vector<torch::Tensor> enc = {x};
  for (int i = 0; i < L_embed_; i++) {
    enc.push_back(torch::sin(std::pow(2.0f, i) * x));
    enc.push_back(torch::cos(std::pow(2.0f, i) * x));
  }
  return torch::cat(enc, -1);
}
