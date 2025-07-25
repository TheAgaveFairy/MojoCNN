from layout import Layout, LayoutTensor, print_layout
from math import sqrt, exp
from random import random_float64, seed
from sys.info import sizeof
from sys import stderr, is_big_endian, argv
from utils.index import IndexList
from time import perf_counter_ns, sleep
import os

from gpu.host import DeviceContext, DeviceFunction, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim, barrier
from layout.tensor_builder import LayoutTensorBuild

import lenet
from lenet import LeNet5, Image
from helpers import readData, showProgress

alias LENGTH_KERNEL = lenet.LENGTH_KERNEL
alias LENGTH_KERNEL_SQ = lenet.LENGTH_KERNEL_SQ

alias LENGTH_FEATURE0 = lenet.LENGTH_FEATURE0
alias LENGTH_FEATURE1 = lenet.LENGTH_FEATURE1
alias LENGTH_FEATURE2 = lenet.LENGTH_FEATURE2
alias LENGTH_FEATURE3 = lenet.LENGTH_FEATURE3
alias LENGTH_FEATURE4 = lenet.LENGTH_FEATURE4
alias LENGTH_FEATURE5 = lenet.LENGTH_FEATURE5

alias INPUT  = lenet.INPUT 
alias LAYER1 = lenet.LAYER1
alias LAYER2 = lenet.LAYER2
alias LAYER3 = lenet.LAYER3
alias LAYER4 = lenet.LAYER4
alias LAYER5 = lenet.LAYER5
alias OUTPUT = lenet.OUTPUT

alias ALPHA = lenet.ALPHA
alias PADDING = lenet.PADDING

alias IMAGE_SIZE = lenet.IMAGE_SIZE
alias PADDED_SIZE = lenet.PADDED_SIZE
alias ftype = lenet.ftype # model's float type, must match "lenet" cpu version because we'll call those constructors

alias COUNT_TRAIN = 60_000
alias COUNT_TEST = 10_000

struct LeNet5GPU():
    """
    The LeNet5 model. In the actual LeCun et al implementation, there is some
    notable sparsity in final layers that is not in this version.
    """
    # WEIGHTS
    alias w0_1_layout = Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)
    var w01_storage: DeviceBuffer[ftype]
    var weight0_1: LayoutTensor[mut = True, ftype, Self.w0_1_layout, MutableAnyOrigin]
    
    alias w2_3_layout = Layout.row_major(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL)
    var w23_storage: DeviceBuffer[ftype]
    var weight2_3: LayoutTensor[mut = True, ftype, Self.w2_3_layout, MutableAnyOrigin]
    
    alias w4_5_layout = Layout.row_major(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL)
    var w45_storage: DeviceBuffer[ftype]
    var weight4_5: LayoutTensor[mut = True, ftype, Self.w4_5_layout, MutableAnyOrigin]
    
    alias w5_6_layout = Layout.row_major(LAYER5 * LENGTH_FEATURE5 *  LENGTH_FEATURE5, OUTPUT)
    var w56_storage: DeviceBuffer[ftype]
    var weight5_6: LayoutTensor[mut = True, ftype, Self.w5_6_layout, MutableAnyOrigin]

    # BIASES
    alias b0_1_layout = Layout.row_major(LAYER1)
    var b01_storage: DeviceBuffer[ftype]
    var bias0_1: LayoutTensor[mut = True, ftype, Self.b0_1_layout, MutableAnyOrigin]
    
    alias b2_3_layout = Layout.row_major(LAYER3)
    var b23_storage: DeviceBuffer[ftype]
    var bias2_3: LayoutTensor[mut = True, ftype, Self.b2_3_layout, MutableAnyOrigin]
    
    alias b4_5_layout = Layout.row_major(LAYER5)
    var b45_storage: DeviceBuffer[ftype]
    var bias4_5: LayoutTensor[mut = True, ftype, Self.b4_5_layout, MutableAnyOrigin]
    
    alias b5_6_layout = Layout.row_major(OUTPUT)
    var b56_storage: DeviceBuffer[ftype]
    var bias5_6: LayoutTensor[mut = True, ftype, Self.b5_6_layout, MutableAnyOrigin]

    fn __init__(out self) raises:
        """
        Initialize to all zeros, for training you'll want to randomizeWeights(),
        or for inference, read in from a file. Only biases really need to be set
        to zeroes.
        """
        try:
            with DeviceContext() as ctx:
                self.w01_storage = ctx.enqueue_create_buffer[ftype](Self.w0_1_layout.size()).enqueue_fill(0)
                self.w23_storage = ctx.enqueue_create_buffer[ftype](Self.w2_3_layout.size()).enqueue_fill(0)
                self.w45_storage = ctx.enqueue_create_buffer[ftype](Self.w4_5_layout.size()).enqueue_fill(0)
                self.w56_storage = ctx.enqueue_create_buffer[ftype](Self.w5_6_layout.size()).enqueue_fill(0)

                # BIASES, no more .stack_allocation()
                self.b01_storage = ctx.enqueue_create_buffer[ftype](Self.b0_1_layout.size()).enqueue_fill(0)
                self.b23_storage = ctx.enqueue_create_buffer[ftype](Self.b2_3_layout.size()).enqueue_fill(0)
                self.b45_storage = ctx.enqueue_create_buffer[ftype](Self.b4_5_layout.size()).enqueue_fill(0)
                self.b56_storage = ctx.enqueue_create_buffer[ftype](Self.b5_6_layout.size()).enqueue_fill(0)

                ctx.synchronize()
                
                self.weight0_1 = __type_of(self.weight0_1)(self.w01_storage)
                self.weight2_3 = __type_of(self.weight2_3)(self.w23_storage)
                self.weight4_5 = __type_of(self.weight4_5)(self.w45_storage)
                self.weight5_6 = __type_of(self.weight5_6)(self.w56_storage)
                
                self.bias0_1 = __type_of(self.bias0_1)(self.b01_storage)
                self.bias2_3 = __type_of(self.bias2_3)(self.b23_storage)
                self.bias4_5 = __type_of(self.bias4_5)(self.b45_storage)
                self.bias5_6 = __type_of(self.bias5_6)(self.b56_storage)
                
        except e:
            print("Something went wrong intializing LeNet5GPU", e)
            #self.weight0_1 = __type_of(self.weight0_1)(UnsafePointer[Scalar[ftype]].alloc(1))
            raise e
        
    fn __init__(out self, cpu_model: LeNet5) raises:
        try:
            with DeviceContext() as ctx:
                #print("Allocating LeNet5 from CPU version to GPU", ctx.name())
                # enqueue fill probably could instead be some form of "unitialized = True"
                self.w01_storage = ctx.enqueue_create_buffer[ftype](Self.w0_1_layout.size()).enqueue_fill(0)
                self.w01_storage.enqueue_copy_from(cpu_model.weight0_1.ptr)
                
                self.w23_storage = ctx.enqueue_create_buffer[ftype](Self.w2_3_layout.size()).enqueue_fill(0)
                self.w23_storage.enqueue_copy_from(cpu_model.weight2_3.ptr)
                
                self.w45_storage = ctx.enqueue_create_buffer[ftype](Self.w4_5_layout.size()).enqueue_fill(0)
                self.w45_storage.enqueue_copy_from(cpu_model.weight4_5.ptr)
                
                self.w56_storage = ctx.enqueue_create_buffer[ftype](Self.w5_6_layout.size()).enqueue_fill(0)
                self.w56_storage.enqueue_copy_from(cpu_model.weight5_6.ptr) 

                # BIASES, no more .stack_allocation()
                self.b01_storage = ctx.enqueue_create_buffer[ftype](Self.b0_1_layout.size()).enqueue_fill(0)
                self.b01_storage.enqueue_copy_from(cpu_model.bias0_1.ptr)
                
                self.b23_storage = ctx.enqueue_create_buffer[ftype](Self.b2_3_layout.size()).enqueue_fill(0)
                self.b23_storage.enqueue_copy_from(cpu_model.bias2_3.ptr)
                
                self.b45_storage = ctx.enqueue_create_buffer[ftype](Self.b4_5_layout.size()).enqueue_fill(0)
                self.b45_storage.enqueue_copy_from(cpu_model.bias4_5.ptr)
                
                self.b56_storage = ctx.enqueue_create_buffer[ftype](Self.b5_6_layout.size()).enqueue_fill(0)
                self.b56_storage.enqueue_copy_from(cpu_model.bias5_6.ptr)
                
                ctx.synchronize()
                
                self.weight0_1 = __type_of(self.weight0_1)(self.w01_storage)
                self.weight2_3 = __type_of(self.weight2_3)(self.w23_storage)
                self.weight4_5 = __type_of(self.weight4_5)(self.w45_storage)
                self.weight5_6 = __type_of(self.weight5_6)(self.w56_storage)

                self.bias0_1 = __type_of(self.bias0_1)(self.b01_storage)
                self.bias2_3 = __type_of(self.bias2_3)(self.b23_storage)
                self.bias4_5 = __type_of(self.bias4_5)(self.b45_storage)
                self.bias5_6 = __type_of(self.bias5_6)(self.b56_storage)
                
        except e:
            print("Error intializing LeNet5GPU", e)
            raise e

struct FeatureGPU(Copyable, Movable):
    """
    Holds intermediate results on the GPU.
    """
    alias input_layout = Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0)
    var input_storage: DeviceBuffer[ftype]
    var input: LayoutTensor[mut = True, ftype, FeatureGPU.input_layout, MutableAnyOrigin]

    alias layer1_layout = Layout.row_major(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1)
    var layer1_storage: DeviceBuffer[ftype]
    var layer1: LayoutTensor[mut = True, ftype, FeatureGPU.layer1_layout, MutableAnyOrigin]

    alias layer2_layout = Layout.row_major(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2)
    var layer2_storage: DeviceBuffer[ftype]
    var layer2: LayoutTensor[mut = True, ftype, FeatureGPU.layer2_layout, MutableAnyOrigin]

    alias layer3_layout = Layout.row_major(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3)
    var layer3_storage: DeviceBuffer[ftype]
    var layer3: LayoutTensor[mut = True, ftype, FeatureGPU.layer3_layout, MutableAnyOrigin]
    
    alias layer4_layout = Layout.row_major(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4)
    var layer4_storage: DeviceBuffer[ftype]
    var layer4: LayoutTensor[mut = True, ftype, FeatureGPU.layer4_layout, MutableAnyOrigin]
    
    alias layer5_layout = Layout.row_major(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5)
    var layer5_storage: DeviceBuffer[ftype]
    var layer5: LayoutTensor[mut = True, ftype, FeatureGPU.layer5_layout, MutableAnyOrigin]
    
    alias output_layout = Layout.row_major(OUTPUT)
    var output_storage: DeviceBuffer[ftype]
    var output: LayoutTensor[mut = True, ftype, FeatureGPU.output_layout, MutableAnyOrigin]

    fn __init__(out self) raises:
        """
        Needs to start as all zeros.
        """
        try:
            with DeviceContext() as ctx:
                self.input_storage = ctx.enqueue_create_buffer[ftype](Self.input_layout.size()).enqueue_fill(0)
                self.layer1_storage = ctx.enqueue_create_buffer[ftype](Self.layer1_layout.size()).enqueue_fill(0)
                self.layer2_storage = ctx.enqueue_create_buffer[ftype](Self.layer2_layout.size()).enqueue_fill(0)
                self.layer3_storage = ctx.enqueue_create_buffer[ftype](Self.layer3_layout.size()).enqueue_fill(0)
                self.layer4_storage = ctx.enqueue_create_buffer[ftype](Self.layer4_layout.size()).enqueue_fill(0)
                self.layer5_storage = ctx.enqueue_create_buffer[ftype](Self.layer5_layout.size()).enqueue_fill(0)
                self.output_storage = ctx.enqueue_create_buffer[ftype](Self.output_layout.size()).enqueue_fill(0)
                
                ctx.synchronize()

                self.input = __type_of(self.input)(self.input_storage)
                self.layer1 = __type_of(self.layer1)(self.layer1_storage)
                self.layer2 = __type_of(self.layer2)(self.layer2_storage)
                self.layer3 = __type_of(self.layer3)(self.layer3_storage)
                self.layer4 = __type_of(self.layer4)(self.layer4_storage)
                self.layer5 = __type_of(self.layer5)(self.layer5_storage)
                self.output = __type_of(self.output)(self.output_storage)
        except e:
            print(e)
            raise e
    
    fn __copyinit__(out self, other: Self):
        self.input = other.input
        self.layer1 = other.layer1
        self.layer2 = other.layer2
        self.layer3 = other.layer3
        self.layer4 = other.layer4
        self.layer5 = other.layer5
        self.output = other.output

        self.input_storage = other.input_storage
        self.layer1_storage = other.layer1_storage
        self.layer2_storage = other.layer2_storage
        self.layer3_storage = other.layer3_storage
        self.layer4_storage = other.layer4_storage
        self.layer5_storage = other.layer5_storage
        self.output_storage = other.output_storage

    fn __moveinit__(out self, owned existing: Self):
        self.input = existing.input
        self.layer1 = existing.layer1
        self.layer2 = existing.layer2
        self.layer3 = existing.layer3
        self.layer4 = existing.layer4
        self.layer5 = existing.layer5
        self.output = existing.output

        self.input_storage = existing.input_storage^
        self.layer1_storage = existing.layer1_storage^
        self.layer2_storage = existing.layer2_storage^
        self.layer3_storage = existing.layer3_storage^
        self.layer4_storage = existing.layer4_storage^
        self.layer5_storage = existing.layer5_storage^
        self.output_storage = existing.output_storage^
    
    fn loadInput(self, image: Image) -> None:
        try:
            var normed = image.toNormalized() # (32, 32) -> (1, 32, 32)
            with self.input_storage.map_to_host() as load_me:
                for i in range(normed.shape[0]()): # PADDED_SIZE
                    for j in range(normed.shape[1]()): # PADDED_SIZE
                        load_me[i * PADDED_SIZE + j] = rebind[Scalar[ftype]](normed[i, j])
            image.pixels.ptr.free()
        except e:
            print("loadInput FeatureGPU ERROR", e)
            #raise e

fn reLu(x: Scalar[ftype]) -> Scalar[ftype]:
    return x if x > 0 else 0

fn maxPool1Kernel[count: UInt](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, count]) -> None:
    """
    Runs as block_dim = 28, 28, grid_dim = (count, LAYER1 # channels). We have
    the "extra" threads to make pulling in the global memory to local more
    fasterer, and then just use "every other" thread to do the actual pooling.
    """
    var img_idx = block_idx.x # range(count)
    var chan = block_idx.y # range(LAYER1)
    var row = thread_idx.y # range(LENGTH_FEATURE1)
    var col = thread_idx.x # range(LENGTH_FEATURE1)
    #var flat_idx = row * block_dim.y + col

    alias image_slice = LayoutTensor[mut = True, ftype, Layout.row_major(LENGTH_FEATURE1, LENGTH_FEATURE1), MutableAnyOrigin]
    
    var local_image = image_slice.stack_allocation()
    local_image[row, col] = feats[img_idx].input[chan, row, col]
    barrier()

    # actual pooling
    if row % 2 == 0 and col % 2 == 0:
        var temp: Scalar[ftype] = rebind[Scalar[ftype]](max(local_image[row, col], local_image[row + 1, col]))
        temp = max(temp, rebind[Scalar[ftype]](local_image[row + 1, col + 1]))
        temp = max(temp, rebind[Scalar[ftype]](local_image[row, col + 1]))

        feats[img_idx].layer2[chan, row, col] = temp

fn maxPool1Forward[count: Int](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, count], pool1_kernel: DeviceFunction) raises -> None:
    """
    Probably will become method of LeNet5GPU.
    """
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(pool1_kernel, lenet, feats, grid_dim = (count, LAYER1), block_dim = (LENGTH_FEATURE1, LENGTH_FEATURE1))
            ctx.synchronize()
    except e:
        print(e)

fn conv2FusedKernel[count: UInt, action: fn(Scalar[ftype]) -> Scalar[ftype]](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, count]) -> None:
    """
    Redo this later to be better.
    """
    alias in_chans = LAYER2 # lenet.weight0_1.shape[0]() # INPUT
    alias out_chans = LAYER3 # lenet.weight0_1.shape[1]() # LAYER1
    alias kernel_length = LENGTH_KERNEL # lenet.weight0_1.shape[2]() # == shape[3]
    alias feat_in = LENGTH_FEATURE2 #feats[0].input.shape[1]() # etc. LENGTH_FEATURE0
    alias feat_out = LENGTH_FEATURE3 #feats[0].layer1.shape[1]() # etc. LENGTH_FEATURE 1 == block_dim.x == block_dim.y

    var img_idx = block_idx.x
    var row = thread_idx.y
    var col = thread_idx.x
    var flat_idx = row * block_dim.y + col
    var TPB = block_dim.x * block_dim.y # 100

    # TODO: ERROR IN HERE
    _ = """
    var local_kernels = __type_of(lenet.weight2_3).stack_allocation() # [in_c 6, out_c 16, len_kern 5, and 5] = 2400
    for i in range(2400 // feat_out):
        #local_kernels.ptr[flat_idx + i * feat_out] = 1#lenet.weight2_3.ptr[flat_idx + i * feat_out]
        local_kernels[0,0,0,0] = 1
    barrier()
    """

    var local_image = __type_of(feats[img_idx].layer2).stack_allocation() # ftype layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2]; 6, 14, 14
    var double_worker = feat_in - feat_out # 14 - 10
    @parameter
    for ic in range(in_chans):
        local_image[ic, row, col] = feats[img_idx].layer2[ic, row, col]
        if row < double_worker and col < double_worker:
            local_image[ic, row + feat_out, col + feat_out] = feats[img_idx].layer2[ic, row + feat_out, col + feat_out]

    barrier()

    if row < feat_out and col < feat_out: # should be given
        @parameter
        for oc in range(out_chans):
            var result: Scalar[ftype] = 0
            @parameter
            for ic in range(in_chans):
                # VALID CONVOLUTION HERE
                @parameter
                for i in range(kernel_length):    
                    @parameter
                    for j in range(kernel_length):
                        var in_row = row + i
                        var in_col = col + j

                        #result += rebind[Scalar[ftype]](local_image[ic, in_row, in_col]) * rebind[Scalar[ftype]](local_kernels[ic, oc, i, j])
                        #result += rebind[Scalar[ftype]](feats[img_idx].layer2[ic, in_row, in_col]) * rebind[Scalar[ftype]](lenet.weight2_3[ic, oc, i, j])
                        result += rebind[Scalar[ftype]](local_image[ic, in_row, in_col]) * rebind[Scalar[ftype]](lenet.weight2_3[ic, oc, i, j])

                feats[img_idx].layer3[oc, row, col] = action(rebind[Scalar[ftype]](result + lenet.bias2_3[oc])) # fused action/reLu and bias


fn conv2Forward[count: Int](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, count], conv2_kernel: DeviceFunction) raises -> None:
    """
    Probably will become method of LeNet5GPU.
    Takes in FeatureGPUs so we can access their buffers, and an already compiled kernel to run.
    """
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(conv2_kernel, lenet, feats, grid_dim = (count), block_dim = (LENGTH_FEATURE3, LENGTH_FEATURE3))
            ctx.synchronize()
    except e:
        print(e)

fn conv1FusedKernel[count: UInt, action: fn(Scalar[ftype]) -> Scalar[ftype]](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, count]) -> None:
    alias in_chans = INPUT # lenet.weight0_1.shape[0]() # INPUT
    alias out_chans = LAYER1 # lenet.weight0_1.shape[1]() # LAYER1
    alias kernel_length = LENGTH_KERNEL # lenet.weight0_1.shape[2]() # == shape[3]
    alias feat_in = LENGTH_FEATURE0 #feats[0].input.shape[1]() # etc. LENGTH_FEATURE0
    alias feat_out = LENGTH_FEATURE0 - LENGTH_KERNEL + 1 #feats[0].layer1.shape[1]() # etc. LENGTH_FEATURE 1 == block_dim.x == block_dim.y

    var img_idx = block_idx.x
    var row = thread_idx.y
    var col = thread_idx.x
    var flat_idx = row * block_dim.y + col

    var local_kernels = __type_of(lenet.weight0_1).stack_allocation() # [in_chan 1, out_chan 6, len_kern 5, len_kern 5]
    if flat_idx < local_kernels.size():
        local_kernels.ptr[flat_idx] = lenet.weight0_1.ptr[flat_idx]

    barrier()

    var local_image = __type_of(feats[img_idx].input).stack_allocation()
    var double_worker = feat_in - feat_out
    local_image[0, row, col] = feats[img_idx].input[0, row, col] # gets 0..28 for x and y
    if col < double_worker and row < double_worker:
        local_image[0, row + feat_out, col + feat_out] = feats[img_idx].input[0, row + feat_out, col + feat_out]

    barrier()

    if row < feat_out and col < feat_out:
        @parameter
        for oc in range(out_chans):
            var result: Scalar[ftype] = 0
            @parameter
            for ic in range(in_chans):
                # VALID CONVOLUTION HERE
                @parameter
                for i in range(kernel_length):    
                    @parameter
                    for j in range(kernel_length):
                        var in_row = row + i
                        var in_col = col + j

                        result += rebind[Scalar[ftype]](local_image[ic, in_row, in_col]) * rebind[Scalar[ftype]](local_kernels[ic, oc, i, j])

                feats[img_idx].layer1[oc, row, col] = action(rebind[Scalar[ftype]](result + lenet.bias0_1[oc])) # fused action/reLu and bias

fn conv1Forward[count: Int](lenet: LeNet5GPU, feats: InlineArray[FeatureGPU, count], conv1_kernel: DeviceFunction) raises -> None:
    """
    Probably will become method of LeNet5GPU.
    Takes in FeatureGPUs so we can access their buffers, and an already compiled kernel to run.
    """
    try:
        with DeviceContext() as ctx:
            ctx.enqueue_function(conv1_kernel, lenet, feats, grid_dim = (count), block_dim = (LENGTH_FEATURE1, LENGTH_FEATURE1))
            ctx.synchronize()
    except e:
        print(e)

fn batchedForward[count: UInt, batch_size: UInt](data: UnsafePointer[Image], model: LeNet5GPU, conv1: DeviceFunction, pool1: DeviceFunction, conv2: DeviceFunction) raises -> None:
    constrained[count % batch_size == 0, "count % batch_size != 0"]()
    try:
        with DeviceContext() as ctx:
            for i in range(0, count, batch_size):
                var features = InlineArray[FeatureGPU, batch_size](fill = FeatureGPU())
                for j in range(batch_size):
                    features[j].loadInput(data[i + j])
                conv1Forward(model, features, conv1)
                maxPool1Forward(model, features, pool1)
                conv2Forward(model, features, conv2)
                
                _ = """
                with features[0].layer1_storage.map_to_host() as test:
                    for i in range(features[0].layer1_layout.size()):
                        print(test[i], end = ", ")
                print("\n\n")
                """
                
    except e:
        print("batchedForward ERROR", e)
        raise e

def main():
    var modelCPU = LeNet5.fromFile[DType.float64]("model_f64.dat") # CRASHES???
    print(modelCPU.bias0_1)
    var modelGPUfromCPU = LeNet5GPU(modelCPU)

    print(LENGTH_KERNEL)
    print(LENGTH_FEATURE0, LENGTH_FEATURE1, LENGTH_FEATURE2, LENGTH_FEATURE3, LENGTH_FEATURE4, LENGTH_FEATURE5)
    print(INPUT, LAYER1, LAYER2, LAYER3, LAYER4, LAYER5, OUTPUT)

    var train_data = UnsafePointer[Image].alloc(COUNT_TRAIN)
    var test_data = UnsafePointer[Image].alloc(COUNT_TEST)
    readData(COUNT_TRAIN, "train", train_data)
    readData(COUNT_TEST, "test", test_data)

    try:
        with DeviceContext() as ctx:
            alias batch_size = 50 # more than ~75 fails "uses too much parameter space"

            _ = """
            with modelGPUfromCPU.b01_storage.map_to_host() as test:
                for i in range(6):
                    print(test[i])
            """

            #print("Compiling conv1 kernel...", end = " ")
            var conv1 = ctx.compile_function[conv1FusedKernel[batch_size, reLu]]()
            var pool1 = ctx.compile_function[maxPool1Kernel[batch_size]]()
            var conv2 = ctx.compile_function[conv2FusedKernel[batch_size, reLu]]()
            batchedForward[COUNT_TEST, batch_size](test_data, modelGPUfromCPU, conv1, pool1, conv2)
            #print("done")
    except e:
        print("ERROR IN MAIN", e)
        raise e
        # GOD FORBID YOU EVER WRITE "RAISE" WITHOUT THE EXCEPTION NAME FOLLOWING IT, ASDHGASIDHFASODHIVCABNS
