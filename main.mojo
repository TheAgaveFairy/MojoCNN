from random import seed
from sys.info import num_logical_cores
from sys import stderr
from time import perf_counter_ns
import os
import benchmark
from gpu.host import DeviceContext

from image import Image
from lenet import LeNet5, ftype, training, testing
from lenetgpu import (
    LeNet5GPU,
    conv1FusedKernel,
    conv2FusedKernel,
    conv3FusedKernel,
    maxPool1Kernel,
    maxPool2Kernel,
    matMulFusedKernel,
    batchedForward,
)

from helpers import showProgress, reLu
from dataloader import MNISTDataRepository
from resultlogger import MultiFileLogger

# note this technically isn't LeNet5 as some of the final connections are full instead of sparse, see their paper
# the penultimate layer of size 84 isnt included either, see their paper

alias COUNT_TRAIN = MNISTDataRepository.COUNT_TRAIN
alias COUNT_TEST = MNISTDataRepository.COUNT_TEST


def main():
    print("CPU Testing")  # , num_logical_cores())
    var data_repo = MNISTDataRepository()
    var logger = MultiFileLogger("results/")

    var train_data = UnsafePointer[Image].alloc(COUNT_TRAIN)
    var test_data = UnsafePointer[Image].alloc(COUNT_TEST)

    var batch_sizes = [100]  # , 300, 600, 1000]
    print(len(batch_sizes), "Batch size test[s] to run")
    for b_sz in batch_sizes:  # range(tests_to_run):
        print("\tBatch size:", b_sz)
        seed(0)  # for random, we could search for a better seed for our shuffleData
        # we free the images as we load them into the model so we need to reload
        data_repo.loadTrainingData(COUNT_TRAIN, train_data)
        data_repo.loadTestingData(COUNT_TEST, test_data)
        data_repo.shuffleData(
            train_data, COUNT_TRAIN
        )  # "hope" for a golden ticket

        var model = LeNet5()
        model.randomizeWeights()

        var start_time = perf_counter_ns()
        training(model, train_data, b_sz, COUNT_TRAIN, logger)
        var training_time = perf_counter_ns()
        var elapsed = training_time - start_time
        print(
            "\n\tTraining done in", elapsed // 1_000_000, "ms. Now testing..."
        )

        var correct = testing(model, test_data, COUNT_TEST)
        var end_time = perf_counter_ns()
        elapsed = end_time - training_time
        logger.logInferenceResult("CPU", elapsed, correct, COUNT_TEST, 1, ftype)
        print(
            "\t",
            correct,
            "/",
            COUNT_TEST,
            "correct\n\t",
            elapsed // 1_000_000,
            "ms for testing.",
        )
        # TODO: SAVE THE MODEL TO A FILE

    # TESTING A PRETRAINED VERSION FROM OLD FILE

    alias model_name = "models/model_f64.dat"
    alias saved_model_dtype = DType.float64

    print("Loading and testing a saved model: '" + model_name + "'")
    var modelCPU = LeNet5.fromFile[saved_model_dtype](model_name)
    data_repo.loadTrainingData(COUNT_TRAIN, train_data)
    data_repo.loadTestingData(COUNT_TEST, test_data)
    start_time = perf_counter_ns()
    var correct = testing(modelCPU, train_data, COUNT_TRAIN)
    end_time = perf_counter_ns()
    print("\t", correct, "/", COUNT_TRAIN, "correct")
    elapsed = end_time - start_time  # // 1_000_000
    print("\t", elapsed // 1_000_000, "ms")
    logger.logInferenceResult(
        "CPU", elapsed, correct, COUNT_TRAIN, 1, saved_model_dtype
    )

    var modelGPUfromCPU = LeNet5GPU(modelCPU)

    # print("Kernel Length:", LENGTH_KERNEL)
    # print("Feature 0->5:", LENGTH_FEATURE0, LENGTH_FEATURE1, LENGTH_FEATURE2, LENGTH_FEATURE3, LENGTH_FEATURE4, LENGTH_FEATURE5)
    # print("Input Channels, Layer1->5, Output:", INPUT, LAYER1, LAYER2, LAYER3, LAYER4, LAYER5, OUTPUT)

    data_repo.loadTrainingData(COUNT_TRAIN, train_data)
    data_repo.loadTestingData(COUNT_TEST, test_data)

    try:
        with DeviceContext() as ctx:
            var device_name = ctx.name()
            print(
                "\nDevice found:",
                device_name,
                ". Compiling kernels and testing...")
            alias batch_size = 50  # more than ~75 fails "uses too much parameter space"

            var conv1 = ctx.compile_function[
                conv1FusedKernel[batch_size, reLu]
            ]()
            var pool1 = ctx.compile_function[maxPool1Kernel[batch_size]]()
            var conv2 = ctx.compile_function[
                conv2FusedKernel[batch_size, reLu]
            ]()
            var pool2 = ctx.compile_function[maxPool2Kernel[batch_size]]()
            var conv3 = ctx.compile_function[
                conv3FusedKernel[batch_size, reLu]
            ]()
            var matmul = ctx.compile_function[
                matMulFusedKernel[batch_size, reLu]
            ]()

            var start_time = perf_counter_ns()

            var correct = batchedForward[COUNT_TRAIN, batch_size](
                train_data,
                modelGPUfromCPU,
                conv1,
                pool1,
                conv2,
                pool2,
                conv3,
                matmul,
            )
            var end_time = perf_counter_ns()
            var elapsed = end_time - start_time  # // 1_000_000

            print("\t", correct, "/", COUNT_TRAIN, "correct")
            print("\t", elapsed // 1_000_000, "ms")
            logger.logInferenceResult(
                device_name, elapsed, correct, COUNT_TRAIN, batch_size, ftype
            )
    except e:
        print("ERROR IN MAIN", e)
        raise e
        # don't forget to tell "raise" what to raise, compiler doesn't handle that well
