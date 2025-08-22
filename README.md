# MojoCNN
Note that the linear layer of size 84 is missing in this version, for now, for comparisons against an older CUDA version I've done where  https://github.com/fan-wenjie/LeNet-5 was the inspiration.

The model_f64 is a trained version of the model saved from the C project in double format.

GPU kernels are for inference only, for now. Initial findings show that I'm about 4% faster than a PyTorch implementation, surprisingly. This is with a batch size of about 50 on my RTX 3070 (8GB). Without more profiling tools (soon to come, IIRC), I don't think I'd super know where to look to improve that possibility. I tested up to about 75 successfully.

There certainly are plenty of places for improvement. I think some of the API is unstable as well. Some TODOs and open mysteries and upcoming expected changes (InlineArray run_destructors = True). No tiling is done, they have their own methods to load things from global memory into shared, etc.
