# MojoCNN
Working on converting this LeNet5 project to Mojo. Starting with a near 1:1 version just as a starting point, then I'll work on making it better with GPU kernels, SIMD, better data structures, etc.

https://github.com/fan-wenjie/LeNet-5 is the inspiration

The model_f64 is a trained version of the model saved from the C project in double format. I've also included a mnist csv just in case I want to play with that, but I'm using the binary files from the above project.

Some personal notes: I didn't know that I could do some_layout_tensor.ptr[i] to more easily index through an array, it would simplify a lot of areas of code. I might go back later, but for now, there's a lot of dense indexing.
