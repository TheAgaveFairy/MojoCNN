# MojoCNN
This is an implementation of the classic* LeNet-5 Convolutional Neural Network from scratch in Mojo🔥 for CPU and GPU with its MNIST dataset. GPU is inference only for now with a performance increase of about 6x over the CPU. In comparison to PyTorch, my GPU inference is 4% faster! Batch size for testing was 50. Accuracy is maintained, +/- 0.5%.

*Note that the linear layer of size 84 is missing in this version, for now, so that I could compare against an older C / CUDA version I've done (https://github.com/TheAgaveFairy/LeNet-5). The old C (CPU) version is only 1/2x the speed of the Mojo GPU version, i.e. much better than the Mojo CPU version (I have ideas about why that is - mainly I imagine the all model layers being stored contiguously on the stack is a big help).

## Running This
Pixi is suggested by Modular for their projects and I've loved it. Install pixi, "pixi shell" into the directory, and use the standard "mojo main.mojo" or "mojo lenetgpu.mojo" to run the CPU and GPU version respectively or "mojo build" to get an executable. Mojo 25.5.0.dev2025072405.

## Files
deviceinfo.mojo : just spits out some info about your GPU

helpers.mojo : loading MNIST data, CPU training and testing loops, etc

lenet.mojo : CPU implementation of the model struct, feature buffer, and the input Image structs

lenetgpu.mojo : GPU versions of the model and feature buffers and all kernels, plus a main() for testing
main.mojo : CPU only!

model*.dat : trained versions of the flattened weights from the old C project (again, these lack that penultimate layer of size 84)

*-ubyte : MNIST dataset files for the images and the corresponding labels. note that there's some headers (never fully figured out why)

results03.ods : see the third sheet for a summary. Mojo runs -O3 by default, so C versions were run with this flag as well.

## Project Notes
There certainly are plenty of places for improvement! Kernels, data organization, code organization, upcoming expected changes (still a new language). I'm also not sure if I'm leaking memory in some places (I know of a few "inconsequential" places).

At this time, I'm not planning on implementing training on GPU, nor the penultimate linear 84 layer.

### TODO
I think I could probably consider doing the final argmax on GPU as a part of the final kernel and write results to a buffer of InlineArray[UInt8, batch_size] directly.

I am considering seeing if I can move the CPU buffers from heap to stack, but the eager destruction made that tricky early on when I was still new so I just threw things into UnsafePointers (malloc() equivalent, more or less) and managed memory myself.

Investigate tiling, streams, using built-in methods for loading data to and from sections of memory (CPU, GPU global, GPU shared), implement vectorized / SIMD for some CPU operations, allow saving my trained model to file, profiling so I can find out how to increase batch size beyond 75, better built in comparisons for CPU vs GPU and results logging, use arg parsing to set batch size etc., allow for arbitrary batch\_sizes, the std from_bytes call got "fixed" in a bug fix and I cannot figure out how to use it now which still irks me, figure out the warnings about unused assignments in some moveinits, make the activation function a parameter for the CPU version so its easily passed around, move kernels / forward / dataloading into the respective structs, properly deprecate or fix GPU support in lenet.mojo, I'm sure there's more...

Additionally, there upcoming expected changes (InlineArray run_destructors = True will become default, closures will eventually allow for parameters which would make loading from file easier). I think there was discussion as well of changing some names for calls that "enqueue" asynchronously buffer creation and movement to and from the host, so those names might change?
