from layout import Layout, LayoutTensor, print_layout
#from layout.layout_tensor import LayoutTensorIter
from math import sqrt, exp
from random import random_float64, seed
from sys.info import sizeof
from sys import stderr, is_big_endian
from utils.index import IndexList
from time import perf_counter_ns
import os
import benchmark

from lenet import LeNet5, Feature, Image, loadInput, loadTarget, forward, backward, argMax, ftype, predict, ALPHA
import lenetgpu

# note this technically isn't LeNet5 as some of the final connections are full instead of sparse, see their paper
# and the penultimate layer of size 84 isnt their either

alias FILE_TRAIN_IMAGE =    "train-images-idx3-ubyte"
alias FILE_TRAIN_LABEL =    "train-labels-idx1-ubyte"
alias FILE_TEST_IMAGE =     "t10k-images-idx3-ubyte"
alias FILE_TEST_LABEL =     "t10k-labels-idx1-ubyte"
alias LENET_FILE =          "model.dat"
alias NUM_WEIGHTS =     51902 # can be calculated but we're just hardcoding for some easier checks at load/save
alias COUNT_TRAIN =     60000
alias COUNT_TEST =      10000

alias device = "cpu" # TODO: deprecate this

fn readData(count: Int, test_set: String, ptr: UnsafePointer[Image]):
    """
    Reads in data from a file. I could probably attach this as a method
    for the Image struct.
    """
    #print("Reading images in from", test_set)
    var data_filename: String = FILE_TEST_IMAGE
    var label_filename: String = FILE_TEST_LABEL
    if test_set == "train":
        data_filename = FILE_TRAIN_IMAGE
        label_filename = FILE_TRAIN_LABEL
    try:
        var data_file = open(data_filename, "r")
        var label_file = open(label_filename, "r")
    
        _ = data_file.seek(16, os.SEEK_SET)    # there is some header...
        _ = label_file.seek(8, os.SEEK_SET)  # ... just do this

        alias buffer_size = Image.PixelLayout.size() #IMAGE_SIZE * IMAGE_SIZE
        var image_buffer = UnsafePointer[UInt8].alloc(buffer_size)
        
        for c in range(count): # TODO: this could probably be vectorized
            var data_list = data_file.read_bytes(buffer_size)
            
            var temp = label_file.read_bytes(1)
            var data_label: UInt8 = temp[0]

            @parameter
            for i in range(Image.PixelTensor.shape[0]()):
                @parameter
                for j in range(Image.PixelTensor.shape[1]()):
                    idx = i * Image.PixelTensor.shape[0]() + j
                    image_buffer[idx] = data_list[idx]

            var test_image = Image(image_buffer, data_label)
            ptr[c] = test_image

        data_file.close()
        label_file.close()
        image_buffer.free()
    except e:
        print("Error with input binary files")

fn trainBatch(mut model: LeNet5, inputs: UnsafePointer[Image], batch_size: Int) -> UInt:
    # TODO: Probably could be a method of LeNet5. "correct" ultimately unused
    var buffer = LeNet5()
    var correct = 0

    for i in range(batch_size):
        var feat = Feature()
        var errors = Feature()
        var deltas = LeNet5()
        loadInput(feat, inputs[i])
        forward[device](model, feat)
        var pred = argMax(feat.output)
        var the_label = Int(inputs[i].label)
        if pred == the_label:
            correct += 1

        loadTarget(feat, errors, the_label)
        backward(model, deltas, errors, feat)
        buffer.accumulateFromOther(deltas, 1.0)

    var k: Scalar[ftype] = Scalar[ftype](ALPHA) / batch_size
    model.accumulateFromOther(buffer, k)

    #_ = correct
    return correct

fn train(mut model: LeNet5, input: Image, label: Int):
    # TODO: UNUSED
    pass
    var feat = Feature()
    var errors = Feature()
    var deltas = LeNet5()

    loadInput(feat, input)
    forward[device](model, feat)
    loadTarget(feat, errors, label)
    backward(model, deltas, errors, feat)
    
    model.accumulateFromOther(deltas, ALPHA)

fn training(mut model: LeNet5, data: UnsafePointer[Image], batch_size: Int, total_size: Int):
    print("Training")
    for i in range(0, total_size, batch_size):
        showProgress(i, total_size)
        _ = trainBatch(model, data + i, batch_size) # could store correct

fn testing(model: LeNet5, data: UnsafePointer[Image], total_size: Int) -> Int:
    var correct = 0
    for i in range(total_size):
        showProgress(i, total_size)
        var pred = predict[device](model, data[i])
        var actual = Int(data[i].label)
        correct += 1 if pred == actual else 0

    return correct

fn shuffleData(data: UnsafePointer[Image], count: Int, seed: Int = 69):
    """
    Not needed, but I / Claude wrote it just to play around and learn.
    """
    if count < 1:
        return
    var rng_state = seed
    # Claude 4 gave this rng_state idea
    for i in range(count - 1, 0, -1):
        rng_state = (rng_state * 1664525 + 1013904223) % 2147483647
        var j = Int(rng_state) % (i + 1)

        var temp = data[i]
        data[i] = data[j]
        data[j] = temp

fn showProgress(progress: Int, total: Int) -> None:
    alias bar_width = 50
    var ratio = progress / total
    var filled = Int(bar_width * ratio)
    #print(chr(27) + "[2J",end="")
    print("\r[", end = "")
    for _ in range(filled):
        print("=", end = "")
    for _ in range(filled, bar_width):
        print(" ", end = "")
    print("]", round(ratio * 100, 3), "%", end = "")

def main():
    var train_data = UnsafePointer[Image].alloc(COUNT_TRAIN)
    var test_data = UnsafePointer[Image].alloc(COUNT_TEST)

    var batch_sizes = [100]#, 300, 600, 1000]
    print(len(batch_sizes), "batch_size tests to run")
    for b_sz in batch_sizes: #range(tests_to_run):
        print("\tbatch size:", b_sz)
        seed(0) #random
        # we free the images as we load them into the model so we need to reload
        readData(COUNT_TRAIN, "train", train_data)
        readData(COUNT_TEST, "test", test_data)
        shuffleData(train_data, COUNT_TRAIN) # can set the seed to something "better"

        var model = LeNet5()
        model.randomizeWeights()

        var start_time = perf_counter_ns()
        training(model, train_data, b_sz, COUNT_TRAIN)
        var end_time = perf_counter_ns()
        var elapsed = end_time - start_time

        var correct = testing(model, test_data, COUNT_TEST)
        print("\n\t", correct, "/", COUNT_TEST, "correct\n\t", (elapsed // 1_000_000), "ms\n\t\t")
        # TODO: SAVE THE MODEL TO A FILE
    
    # TESTING A PRETRAINED VERSION FROM OLD FILE

    var model_name = "model_f64.dat"
    print("loading and testing a saved model:", model_name)
    var model = LeNet5.fromFile[DType.float64](model_name)
    readData(COUNT_TRAIN, "train", train_data)
    readData(COUNT_TEST, "test", test_data)
    start_time = perf_counter_ns()
    var correct = testing(model, train_data, COUNT_TRAIN)
    end_time = perf_counter_ns()
    print("\n\t", correct, "/", COUNT_TRAIN, "correct")
    elapsed = end_time - start_time
    print("\t", elapsed , "ns")

    # OS will get these
    #train_data.free()
    #test_data.free()
