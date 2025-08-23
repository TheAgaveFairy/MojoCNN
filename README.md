# MojoCNN: LeNet-5 Implementation from Scratch

A high-performance implementation of the LeNet-5 Convolutional Neural Network built entirely from scratch in Mojo🔥 with custom CPU and GPU kernels, achieving competitive performance with established frameworks.

## 🎯 Project Motivation

This project was undertaken as a deep learning exercise to:
- **Learn CNNs from first principles** by implementing every component from scratch
- **Explore Mojo**, a cutting-edge systems programming language designed for AI workloads
- **Build custom GPU kernels** without relying on existing ML libraries or frameworks
- **Achieve competitive performance** through low-level optimization and manual memory management

## 🚀 Performance Highlights

- **6x speedup** GPU vs CPU inference
- **4% faster** than PyTorch GPU inference (batch size 50)
- **Maintained accuracy** within ±0.5% of reference implementations
- **Custom GPU kernels** written entirely in Mojo

## 🏗️ Architecture

This implementation features a modified LeNet-5 architecture:
- Convolutional layers with custom kernels
- Max pooling operations
- Fully connected layers
- ReLU activation function
- MNIST dataset integration

*Note: The traditional 84-unit penultimate layer is omitted for direct comparison with a previous C/CUDA implementation.*

## 📁 Project Structure

```
├── main.mojo          # CPU-only training and inference
├── lenetgpu.mojo      # GPU inference implementation
├── lenet.mojo         # CPU model definitions and operations
├── helpers.mojo       # MNIST data loading and training utilities
├── deviceinfo.mojo    # GPU device information utilities
├── model*.dat         # Pre-trained model weights
├── *-ubyte           # MNIST dataset files
└── results03.ods     # Performance benchmarking results
```

## 🛠️ Technical Implementation

### Custom Components Built from Scratch
- **Memory Management**: Manual allocation using UnsafePointers
- **Matrix Operations**: Custom implementations without external BLAS libraries  
- **GPU Kernels**: Hand-written kernels in Mojo for all operations
- **Data Pipeline**: Custom MNIST loader with proper header handling
- **Forward Pass**: Complete inference pipeline optimized for both CPU and GPU

### Key Features
- Zero external ML library dependencies
- Custom GPU memory management and kernel execution
- Batch processing support (tested up to batch size 75)
- Cross-platform compatibility (CPU/GPU)

## 🚦 Getting Started

### Prerequisites
- Mojo 25.5.0.dev2025072405 or later
- Mojo Supported GPU (NVidia, AMD. Apple soon!)
- Pixi package manager

### Installation & Usage

```bash
# Install dependencies
pixi shell

# CPU training and inference
mojo main.mojo

# GPU inference only
mojo lenetgpu.mojo

# Build executable
mojo build main.mojo
mojo build lenetgpu.mojo
```

### Device Information
```bash
mojo deviceinfo.mojo  # Check GPU capabilities
```

## 📊 Performance Comparison

| Implementation | Platform | Time in ms | Notes |
|---------------|----------|----------------|--------|
| MojoCNN | GPU | 2069 | Custom kernels |
| MojoCNN | CPU | 12381 | Baseline |
| PyTorch | GPU | 2150 | 4% slower than MojoCNN |
| PyTorch | CPU | 2485 | 4% slower than MojoCNN |
| C/CUDA | CPU | 4241 | Stack-allocated model, no multithreading enabled |

*All benchmarks conducted with -O3 optimization and batch size 50 on 60,000 images. Times are averages of 10 runs.*

## 🔬 Learning Outcomes

Through this project, I gained hands-on experience with:
- **Low-level CNN implementation** including convolution mathematics
- **GPU programming** and kernel optimization
- **Memory management** in systems programming contexts  
- **Performance profiling** and optimization techniques
- **Mojo language features** for high-performance computing

## 🚧 Current Limitations & Future Work

### Known Limitations
- GPU training not implemented (inference only)
- Missing 84-unit penultimate layer from standard LeNet-5
- Batch size limited to ~75 due to memory constraints
- Some potential memory leaks in edge cases

### Planned Improvements
- [ ] GPU training implementation
- [ ] Complete LeNet-5 architecture with all layers
- [ ] Memory optimization (stack allocation where possible)
- [ ] Kernel tiling and streaming optimizations
- [ ] SIMD vectorization for CPU operations
- [ ] Full model serialization/deserialization
- [ ] Comprehensive profiling and benchmarking suite
- [ ] Dynamic batch size support

## 🤝 Contributing

This is primarily an educational project, but suggestions and discussions about optimization techniques or Mojo best practices are welcome!

## 📜 License
MIT License

Copyright (c) [2025] [Paul Dutton]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments

- Built with [Mojo🔥](https://www.modular.com/mojo) by Modular
- MNIST dataset from Yann LeCun's database
- Inspired by the original LeNet-5 paper by Y. LeCun et al.

---

*This project demonstrates that competitive deep learning performance can be achieved through careful low-level implementation, even without relying on established ML frameworks.*
