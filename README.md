<h1 align="center">
  <br>
  <img src="https://raw.githubusercontent.com/rafaelanderka/cNeRF/main/output/example/training.gif" alt="Logo" width="200">
  <br>
  <br>
  <b>cNeRF</b>
  <br>
</h1>

<h4 align="center">A concise C++ implementation of <a href="http://tancik.com/nerf">Neural Radiance Fields (NeRF)</a> using <a href="https://pytorch.org/cppdocs/">LibTorch</a>.</h4>
<br>

## Overview

This project provides a minimal implementation of [Neural Radiance Fields (NERF)](http://tancik.com/nerf), a method for synthesizing novel views of complex scenes using neural inverse modelling. The code is written in C++ and utilizes [LibTorch](https://pytorch.org/cppdocs/) for automatic differentiation.

## Dependencies

- [LibTorch](https://pytorch.org/cppdocs/installing.html) (>= 2.0.0)
- CUDA (>= 11.7, optional)

## Installation

1. Download and install [LibTorch](https://pytorch.org/cppdocs/installing.html) (>= 2.0.0)
2. If you place LibTorch in the project root directory, no additional configuration is required. Alternatively, you can install LibTorch locally and update the CMakeLists.txt file with the appropriate path.
3. Build the project using CMake:

```sh
mkdir build
cd build
cmake ..
make
```

## Usage

After building the project, run the executable with the appropriate command-line arguments to specify the data and output directories:

```sh
./cNeRF /path/to/data /path/to/output
```

## Acknowledgements

This implementation is based on the original [NeRF repository](https://github.com/bmild/nerf) by Mildenhall et al. We thank the authors for their valuable research and open-source code.

## License
This project is licensed under the [MIT License](LICENSE).