from layout import Layout, LayoutTensor, print_layout#, LayoutTensorIter
from layout.layout_tensor import LayoutTensorIter
from math import sqrt, exp
from random import random_float64, seed
from sys.info import sizeof
from sys import stderr
from utils.index import IndexList
#import simd
from time import perf_counter_ns
import os

#note this technically isn't LeNet5 as some of the final connections are full instead of sparse, see their paper

alias FILE_TRAIN_IMAGE =    "train-images-idx3-ubyte"
alias FILE_TRAIN_LABEL =    "train-labels-idx1-ubyte"
alias FILE_TEST_IMAGE =     "t10k-images-idx3-ubyte"
alias FILE_TEST_LABEL =     "t10k-labels-idx1-ubyte"
alias LENET_FILE =          "model.dat"
alias NUM_WEIGHTS =     51902 # can be calculated but we're just hardcoding for some easier checks at load/save
alias COUNT_TRAIN =     60000
alias COUNT_TEST =      10000

alias LENGTH_KERNEL =   5
alias LENGTH_KERNEL_SQ = LENGTH_KERNEL * LENGTH_KERNEL

alias LENGTH_FEATURE0 = 32
alias LENGTH_FEATURE1 = (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
alias LENGTH_FEATURE2 = (LENGTH_FEATURE1 >> 1)
alias LENGTH_FEATURE3 = (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
alias LENGTH_FEATURE4 = (LENGTH_FEATURE3 >> 1)
alias LENGTH_FEATURE5 = (LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

alias INPUT =   1
alias LAYER1 =  6
alias LAYER2 =  6
alias LAYER3 =  16
alias LAYER4 =  16
alias LAYER5 =  120
alias OUTPUT =  10

alias ALPHA = 0.5
alias PADDING = 2

alias IMAGE_SIZE =      28 # as we read it in from the file, its padded to LENGTH_FEATURE0 (a.k.a. PADDED_SIZE)
alias PADDED_SIZE = IMAGE_SIZE + 2 * PADDING # 32 x 32 is what we want eventually # this should equal LENGTH_FEATURE0
alias ftype = DType.float64 # model's float type. pixels are uint8 (bytes), non-negotiable

struct LeNet5(Copyable):
    # WEIGHTS
    alias w0_1_layout = Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)
    var weight0_1: LayoutTensor[mut = True, ftype, LeNet5.w0_1_layout, MutableAnyOrigin]
    
    alias w2_3_layout = Layout.row_major(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL)
    var weight2_3: LayoutTensor[mut = True, ftype, LeNet5.w2_3_layout, MutableAnyOrigin]
    
    alias w4_5_layout = Layout.row_major(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL)
    var weight4_5: LayoutTensor[mut = True, ftype, LeNet5.w4_5_layout, MutableAnyOrigin]
    
    alias w5_6_layout = Layout.row_major(LAYER5 * LENGTH_FEATURE5 *  LENGTH_FEATURE5, OUTPUT)
    var weight5_6: LayoutTensor[mut = True, ftype, LeNet5.w5_6_layout, MutableAnyOrigin]

    # BIASES
    alias b0_1_layout = Layout.row_major(LAYER1)
    var bias0_1: LayoutTensor[mut = True, ftype, LeNet5.b0_1_layout, MutableAnyOrigin]
    
    alias b2_3_layout = Layout.row_major(LAYER3)
    var bias2_3: LayoutTensor[mut = True, ftype, LeNet5.b2_3_layout, MutableAnyOrigin]
    
    alias b4_5_layout = Layout.row_major(LAYER5)
    var bias4_5: LayoutTensor[mut = True, ftype, LeNet5.b4_5_layout, MutableAnyOrigin]
    
    alias b5_6_layout = Layout.row_major(OUTPUT)
    var bias5_6: LayoutTensor[mut = True, ftype, LeNet5.b5_6_layout, MutableAnyOrigin]

    fn __init__(out self):
        var w01_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w0_1_layout.size())
        self.weight0_1 = __type_of(self.weight0_1)(w01_storage).fill(0.0)

        var w23_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w2_3_layout.size())
        self.weight2_3 = __type_of(self.weight2_3)(w23_storage).fill(0.0)
        
        var w45_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w4_5_layout.size())
        self.weight4_5 = __type_of(self.weight4_5)(w45_storage).fill(0.0)
        
        var w56_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w5_6_layout.size())
        self.weight5_6 = __type_of(self.weight5_6)(w56_storage).fill(0.0)

        # BIASES, no more .stack_allocation()
        var b01_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b0_1_layout.size())
        self.bias0_1 = __type_of(self.bias0_1)(b01_storage).fill(0.0)
        
        var b23_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b2_3_layout.size())
        self.bias2_3 = __type_of(self.bias2_3)(b23_storage).fill(0.0)
        
        var b45_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b4_5_layout.size())
        self.bias4_5 = __type_of(self.bias4_5)(b45_storage).fill(0.0)
        
        var b56_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b5_6_layout.size())
        self.bias5_6 = __type_of(self.bias5_6)(b56_storage).fill(0.0)

    fn __copyinit__(out self, other: Self):
        self.weight0_1 = other.weight0_1
        self.weight2_3 = other.weight2_3
        self.weight4_5 = other.weight4_5
        self.weight5_6 = other.weight5_6

        self.bias0_1 = other.bias0_1
        self.bias2_3 = other.bias2_3
        self.bias4_5 = other.bias4_5
        self.bias5_6 = other.bias5_6
    
    fn __del__(owned self):
        self.weight0_1.ptr.free()
        self.weight2_3.ptr.free()
        self.weight4_5.ptr.free()
        self.weight5_6.ptr.free()

        self.bias0_1.ptr.free()
        self.bias2_3.ptr.free()
        self.bias4_5.ptr.free()
        self.bias5_6.ptr.free()

    fn accumulateFromOther(mut self, other: Self, lr: Scalar[ftype]):
        _ = """
        self.weight0_1 += 1.0# lr #other.weight0_1 * lr
        self.weight2_3 += 1.0 # lr #other.weight2_3 * lr 
        self.weight4_5 += 1.0 #other.weight4_5 * lr
        self.weight5_6 += 1.0 #other.weight5_6 * lr

        self.bias0_1 += 1.0 #other.bias0_1 * lr
        self.bias2_3 += 1.0 #other.bias2_3 * lr 
        self.bias4_5 += 1.0 #other.bias4_5 * lr 
        self.bias5_6 += 1.0 #other.bias5_6 * lr
        """
        for i in range(self.weight0_1.shape[0]()):
            for j in range(self.weight0_1.shape[1]()):
                for k in range(self.weight0_1.shape[2]()):
                    for l in range(self.weight0_1.shape[3]()):
                        #print(other.weight0_1[i,j,k,l], lr, end = ", ")
                        self.weight0_1[i,j,k,l] += other.weight0_1[i,j,k,l] * lr

        for i in range(self.weight2_3.shape[0]()):
            for j in range(self.weight2_3.shape[1]()):
                for k in range(self.weight2_3.shape[2]()):
                    for l in range(self.weight2_3.shape[3]()):
                        self.weight2_3[i,j,k,l] += other.weight2_3[i,j,k,l] * lr
        
        for i in range(self.weight4_5.shape[0]()):
            for j in range(self.weight4_5.shape[1]()):
                for k in range(self.weight4_5.shape[2]()):
                    for l in range(self.weight4_5.shape[3]()):
                        self.weight4_5[i,j,k,l] += other.weight4_5[i,j,k,l] * lr

        for i in range(self.weight5_6.shape[0]()):
            for j in range(self.weight5_6.shape[1]()):
                self.weight5_6[i,j] += other.weight5_6[i,j] * lr

        for i in range(self.bias0_1.shape[0]()):
            self.bias0_1[i] += other.bias0_1[i] * lr
        for i in range(self.bias2_3.shape[0]()):
            self.bias2_3[i] += other.bias2_3[i] * lr
        for i in range(self.bias4_5.shape[0]()):
            self.bias4_5[i] += other.bias4_5[i] * lr
        for i in range(self.bias5_6.shape[0]()):
            self.bias5_6[i] += other.bias5_6[i] * lr

    fn randomizeWeights(self):
        for i in range(self.weight0_1.shape[0]()):
            for j in range(self.weight0_1.shape[1]()):
                for k in range(self.weight0_1.shape[2]()):
                    for l in range(self.weight0_1.shape[3]()):
                        self.weight0_1[i,j,k,l] = random_float64(-1.0, 1.0).cast[ftype]()
                        self.weight0_1[i,j,k,l] *= Scalar[ftype](sqrt(6.0 / (LENGTH_KERNEL_SQ * (INPUT + LAYER1))))

        for i in range(self.weight2_3.shape[0]()):
            for j in range(self.weight2_3.shape[1]()):
                for k in range(self.weight2_3.shape[2]()):
                    for l in range(self.weight2_3.shape[3]()):
                        self.weight2_3[i,j,k,l] = random_float64(-1.0, 1.0).cast[ftype]()
                        self.weight2_3[i,j,k,l] *= Scalar[ftype](sqrt(6.0 / (LENGTH_KERNEL_SQ * (LAYER2 + LAYER3))))
        
        for i in range(self.weight4_5.shape[0]()):
            for j in range(self.weight4_5.shape[1]()):
                for k in range(self.weight4_5.shape[2]()):
                    for l in range(self.weight4_5.shape[3]()):
                        self.weight4_5[i,j,k,l] = random_float64(-1.0, 1.0).cast[ftype]()
                        self.weight4_5[i,j,k,l] *= Scalar[ftype](sqrt(6.0 / (LENGTH_KERNEL_SQ * (LAYER4 + LAYER5))))

        for i in range(self.weight5_6.shape[0]()):
            for j in range(self.weight5_6.shape[1]()):
                self.weight5_6[i,j] = random_float64(-1.0, 1.0).cast[ftype]()
                self.weight5_6[i,j] *= Scalar[ftype](sqrt(6.0 / (LAYER5 + OUTPUT)))

        _ = """
        for i in range(self.bias0_1.shape[0]()):
            self.bias0_1[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(self.bias2_3.shape[0]()):
            self.bias2_3[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(self.bias4_5.shape[0]()):
            self.bias4_5[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(self.bias5_6.shape[0]()):
            self.bias5_6[i] = random_float64(-1.0, 1.0).cast[ftype]()
        """

    @staticmethod
    fn bytesToFType[filetype: DType, num_bytes: Int, layout: Layout]
        (bytes: InlineArray[Scalar[DType.uint8], num_bytes],
         tensor: LayoutTensor[mut = True, ftype, layout, MutableAnyOrigin]) -> None:

        alias f_sz = filetype.sizeof()
        alias num_elems = num_bytes / f_sz
        if num_elems != tensor.size() or num_elems != num_bytes / f_sz:
            print("FATAL ERROR CONVERTING BYTES TO TENSOR")
            print("num_elems, tensor.size(), len(bytes):", num_elems, tensor.size(), len(bytes))

        if tensor.layout.rank() == 1: # why can't i use "tensor.rank()"?????
            for idx in range(tensor.size()):
                var i = idx
                
                var buffer = InlineArray[Scalar[DType.uint8], f_sz](uninitialized = True)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[bi] = bytes[temp_idx] # f_sz - 1 - bi to reverse
                var value = SIMD[filetype, 1].from_bytes[](buffer)
                #print(value, end = ", ")
                tensor[i] = value.cast[ftype]()

        if tensor.layout.rank() == 2: # why can't i use "tensor.rank()"?????
            for idx in range(tensor.size()):
                var i = idx // (tensor.shape[1]())
                var j = idx % (tensor.shape[1]())
                
                var buffer = InlineArray[Scalar[DType.uint8], f_sz](uninitialized = True)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[bi] = bytes[temp_idx] # f_sz - 1 - bi to reverse
                    #print("into buffer:", buffer[bi])
                var value = SIMD[filetype, 1].from_bytes[](buffer)
                #print(value, end = ", ")
                tensor[i,j] = value.cast[ftype]()
        
        if tensor.layout.rank() == 3: # why can't i use "tensor.rank()"?????
            for idx in range(tensor.size()):
                var i = idx // (tensor.shape[1]() * tensor.shape[2]())
                var remainder = idx % (tensor.shape[1]() * tensor.shape[2]())
                var j = remainder // tensor.shape[2]()
                var k = remainder % tensor.shape[2]()

                var buffer = InlineArray[Scalar[DType.uint8], f_sz](uninitialized = True)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[bi] = bytes[temp_idx] # f_sz - 1 - bi to reverse
                    #print("into buffer:", buffer[bi])
                var value = SIMD[filetype, 1].from_bytes[](buffer)
                #print(value, end = ", ")
                tensor[i,j,k] = value.cast[ftype]()
                
        if tensor.layout.rank() == 4: # why can't i use "tensor.rank()"?????
            for idx in range(tensor.size()):
                var i = idx // (tensor.shape[1]() * tensor.shape[2]() * tensor.shape[3]())
                var remainder = idx % (tensor.shape[1]() * tensor.shape[2]() * tensor.shape[3]())
                var j = remainder // (tensor.shape[2]() * tensor.shape[3]())
                remainder = remainder % (tensor.shape[2]() * tensor.shape[3]())
                var k = remainder // tensor.shape[3]()
                var l = remainder % tensor.shape[3]()

                var buffer = InlineArray[Scalar[DType.uint8], f_sz](uninitialized = True)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[bi] = bytes[temp_idx] # f_sz - 1 - bi to reverse
                var value = SIMD[filetype, 1].from_bytes[](buffer)
                #print(value, end = ", ")
                tensor[i,j,k,l] = value.cast[ftype]()

    @staticmethod
    fn fromFile[filetype: DType](filename: String) -> Self:
        """
        Closures can't have parameters, yet. TODO
        fn helper[layer_size: Int, layout: Layout](weights: LayoutTensor[mut = True, ftype, layout, MutableAnyOrigin]):
            alias size_of_layer = layout.size()
            alias bytes_to_read = size_of_layer * bytes_per_file_weight
            var bytes: List[UInt8]
            try:
                bytes = model_file.read_bytes(bytes_to_read)
            except ee:
                print("helper fromFile", ee)
            var buffer = InlineArray[Scalar[DType.uint8], bytes_to_read](uninitialized = True)
            for i in range(bytes_to_read):
                buffer[i] = bytes[i]
            Self.bytesToFType[filetype, bytes_to_read, layout](buffer, weights)
        .
        """
        alias bytes_per_file_weight = filetype.sizeof()# TODO sizeof[filetype]() won't work, must use filetype.sizeof() why?????
        #print("Loading LeNet5 from ", filename, ". filetype number of bytes:", bytes_per_file_weight)
        var model = LeNet5()

        try:
            var model_file = open(filename, "r")

            #TODO : MAKE THIS BEHAVIOR FUNCTION OR CLOSURE (SEE DOCSTRING) SO I DON'T HAVE TO COPY/PASTE THIS JUNK
            # I ALMOST MISS C MACROS

            # WEIGHTS
            #print("weight0_1 copying...")
            alias w01_sz = model.w0_1_layout.size()#INPUT * LAYER1 * LENGTH_KERNEL * LENGTH_KERNEL
            alias w01_bytes_to_read = w01_sz * bytes_per_file_weight
            var bytes = model_file.read_bytes(w01_bytes_to_read)
            var w01_buffer = InlineArray[Scalar[DType.uint8], w01_bytes_to_read](uninitialized = True)
            for i in range(w01_bytes_to_read):
                w01_buffer[i] = bytes[i] # could reverse this here, etc
            Self.bytesToFType[filetype, w01_bytes_to_read, model.w0_1_layout](w01_buffer, model.weight0_1)

            #print("weight2_3 copying...")
            alias w23_sz = model.w2_3_layout.size()
            alias w23_bytes_to_read = w23_sz * bytes_per_file_weight
            bytes = model_file.read_bytes(w23_bytes_to_read)
            var w23_buffer = InlineArray[Scalar[DType.uint8], w23_bytes_to_read](uninitialized = True)
            for i in range(w23_bytes_to_read):
                w23_buffer[i] = bytes[i] # could reverse this here, etc
            Self.bytesToFType[filetype, w23_bytes_to_read, model.w2_3_layout](w23_buffer, model.weight2_3)

            #print("weight4_5 copying...")
            alias w45_sz = model.w4_5_layout.size()
            alias w45_bytes_to_read = w45_sz * bytes_per_file_weight
            bytes = model_file.read_bytes(w45_bytes_to_read)
            var w45_buffer = InlineArray[Scalar[DType.uint8], w45_bytes_to_read](uninitialized = True)
            for i in range(w45_bytes_to_read):
                w45_buffer[i] = bytes[i] # could reverse this here, etc
            Self.bytesToFType[filetype, w45_bytes_to_read, model.w4_5_layout](w45_buffer, model.weight4_5)

            #print("weight5_6 copying...")
            alias w56_sz = model.w5_6_layout.size()
            alias w56_bytes_to_read = w56_sz * bytes_per_file_weight
            bytes = model_file.read_bytes(w56_bytes_to_read)
            var w56_buffer = InlineArray[Scalar[DType.uint8], w56_bytes_to_read](uninitialized = True)
            for i in range(w56_bytes_to_read):
                w56_buffer[i] = bytes[i] # could reverse this here, etc
            Self.bytesToFType[filetype, w56_bytes_to_read, model.w5_6_layout](w56_buffer, model.weight5_6)

            # BIASES
            #print("bias0_1 copying...")
            alias b01_sz = model.b0_1_layout.size()#INPUT * LAYER1 * LENGTH_KERNEL * LENGTH_KERNEL
            alias b01_bytes_to_read = b01_sz * bytes_per_file_weight
            bytes = model_file.read_bytes(b01_bytes_to_read)
            var b01_buffer = InlineArray[Scalar[DType.uint8], b01_bytes_to_read](uninitialized = True)
            for i in range(b01_bytes_to_read):
                b01_buffer[i] = bytes[i] # could reverse this here, etc
            Self.bytesToFType[filetype, b01_bytes_to_read, model.b0_1_layout](b01_buffer, model.bias0_1)

            #print("bias2_3 copying...")
            alias b23_sz = model.b2_3_layout.size()
            alias b23_bytes_to_read = b23_sz * bytes_per_file_weight
            bytes = model_file.read_bytes(b23_bytes_to_read)
            var b23_buffer = InlineArray[Scalar[DType.uint8], b23_bytes_to_read](uninitialized = True)
            for i in range(b23_bytes_to_read):
                b23_buffer[i] = bytes[i] # could reverse this here, etc
            Self.bytesToFType[filetype, b23_bytes_to_read, model.b2_3_layout](b23_buffer, model.bias2_3)

            #print("bias4_5 copying...")
            alias b45_sz = model.b4_5_layout.size()
            alias b45_bytes_to_read = b45_sz * bytes_per_file_weight
            bytes = model_file.read_bytes(b45_bytes_to_read)
            var b45_buffer = InlineArray[Scalar[DType.uint8], b45_bytes_to_read](uninitialized = True)
            for i in range(b45_bytes_to_read):
                b45_buffer[i] = bytes[i] # could reverse this here, etc
            Self.bytesToFType[filetype, b45_bytes_to_read, model.b4_5_layout](b45_buffer, model.bias4_5)

            #print("bias5_6 copying...")
            alias b56_sz = model.b5_6_layout.size()
            alias b56_bytes_to_read = b56_sz * bytes_per_file_weight
            bytes = model_file.read_bytes(b56_bytes_to_read)
            var b56_buffer = InlineArray[Scalar[DType.uint8], b56_bytes_to_read](uninitialized = True)
            for i in range(b56_bytes_to_read):
                b56_buffer[i] = bytes[i] # could reverse this here, etc
            Self.bytesToFType[filetype, b56_bytes_to_read, model.b5_6_layout](b56_buffer, model.bias5_6)

        except e:
            print("error at reading lenet5 from file", e)
        return model
        

struct Feature():
    alias input_layout = Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0)
    var input: LayoutTensor[mut = True, ftype, Feature.input_layout, MutableAnyOrigin]

    alias layer1_layout = Layout.row_major(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1)
    var layer1: LayoutTensor[mut = True, ftype, Feature.layer1_layout, MutableAnyOrigin]

    alias layer2_layout = Layout.row_major(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2)
    var layer2: LayoutTensor[mut = True, ftype, Feature.layer2_layout, MutableAnyOrigin]

    alias layer3_layout = Layout.row_major(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3)
    var layer3: LayoutTensor[mut = True, ftype, Feature.layer3_layout, MutableAnyOrigin]
    
    alias layer4_layout = Layout.row_major(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4)
    var layer4: LayoutTensor[mut = True, ftype, Feature.layer4_layout, MutableAnyOrigin]
    
    alias layer5_layout = Layout.row_major(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5)
    var layer5: LayoutTensor[mut = True, ftype, Feature.layer5_layout, MutableAnyOrigin]
    
    alias output_layout = Layout.row_major(OUTPUT)
    var output: LayoutTensor[mut = True, ftype, Feature.output_layout, MutableAnyOrigin]

    fn __init__(out self):
        var input_storage = UnsafePointer[Scalar[ftype]].alloc(Self.input_layout.size())
        self.input = __type_of(self.input)(input_storage).fill(0.0)

        var layer1_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer1_layout.size())
        self.layer1 = __type_of(self.layer1)(layer1_storage).fill(0.0)
        
        var layer2_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer2_layout.size())
        self.layer2 = __type_of(self.layer2)(layer2_storage).fill(0.0)

        var layer3_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer3_layout.size())
        self.layer3 = __type_of(self.layer3)(layer3_storage).fill(0.0)

        var layer4_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer4_layout.size())
        self.layer4 = __type_of(self.layer4)(layer4_storage).fill(0.0)

        var layer5_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer5_layout.size())
        self.layer5 = __type_of(self.layer5)(layer5_storage).fill(0.0)

        var output_storage = UnsafePointer[Scalar[ftype]].alloc(Self.output_layout.size())
        self.output = __type_of(self.output)(output_storage).fill(0.0)

    fn __del__(owned self):
        self.input.ptr.free()
        self.layer1.ptr.free()
        self.layer2.ptr.free()
        self.layer3.ptr.free()
        self.layer4.ptr.free()
        self.layer5.ptr.free()
        self.output.ptr.free()

struct Image(Stringable, Copyable):
    alias PixelLayout = Layout.row_major(IMAGE_SIZE, IMAGE_SIZE)
    alias PixelStorage = InlineArray[UInt8, Self.PixelLayout.size()]#(uninitialized = True)
    alias PixelTensor = LayoutTensor[mut = True, DType.uint8, Self.PixelLayout, MutableAnyOrigin]

    alias DataLayout = Layout.row_major(PADDED_SIZE, PADDED_SIZE)
    alias DataStorage = InlineArray[Scalar[ftype], Self.DataLayout.size()]#
    alias DataTensor = LayoutTensor[mut = True, ftype, Self.DataLayout, MutableAnyOrigin]
    
    var pixels: Self.PixelTensor
    var label: UInt8 # digits [0, 9] MNIST

    fn __init__(out self, ptr: UnsafePointer[UInt8], label: UInt8):
        if label > 9:
            print("Error with incoming label for image:", label)
        #var storage = InlineArray[UInt8, IMAGE_SIZE * IMAGE_SIZE](uninitialized = True)
        var storage = UnsafePointer[UInt8].alloc(Self.PixelLayout.size())
        var temp_pixels = Self.PixelTensor(storage)
        # memcpy probably possible
        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                var idx = r * IMAGE_SIZE + c
                temp_pixels[r, c] = ptr[idx]

        self.pixels = temp_pixels
        self.label = label

    # (Movable)
    #fn __moveinit__(out self, owned existing: Self):
    #    existing.pixels = self.pixels^
    #    existing.label = self.label^
    #fn __del__(owned self): who the fuck does this and when
    #    self.pixels.free()

    fn toNormalized(self) -> Self.DataTensor:
        """
        Normalizes from 28x28 uint8 to zero-padded 32x32 float32 (or whatever ftype is for the model).
        """
        #mut = False gives a terrible terrible compiler warning, please fix, as an aside for making LayoutTensors
        var storage = UnsafePointer[Scalar[ftype]].alloc(Self.DataLayout.size())
        var tensor = Self.DataTensor(storage).fill(0.0)

        var mean: Float64
        var std: Float64

        var sum: UInt64 = 0
        var std_sum: UInt64 = 0
        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                sum += UInt(self.pixels[r, c]) # not taking advantage of SIMD possibly?
                std_sum += Int(self.pixels[r, c].cast[DType.uint64]() * self.pixels[r, c].cast[DType.uint64]()) # rebind[UInt64]
        
        alias num_elems = IMAGE_SIZE * IMAGE_SIZE
        mean = Float64(sum) / num_elems
        var temp = Float64(std_sum) / num_elems - mean * mean
        std = sqrt(temp)

        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                tensor[r + PADDING, c + PADDING] = ((Float64(self.pixels[r, c]) - mean) / std).cast[ftype]()
         
        return tensor # who manages this memory honestly

    fn __str__(self) -> String:
        var temp: String = "Raw From File -> Label: " + String(self.label) + "\n"
        for r in range(self.pixels.shape[0]()): # rows
            for c in range(self.pixels.shape[1]()): # cols
                if self.pixels[r, c] < 32:
                    temp += " "
                elif self.pixels[r, c] < 64:
                    temp += "."
                elif self.pixels[r, c] < 96:
                    temp += ","
                elif self.pixels[r, c] < 128:
                    temp += "o"
                elif self.pixels[r, c] < 160:
                    temp += "x"
                elif self.pixels[r, c] < 192:
                    temp += "$"
                elif self.pixels[r, c] < 224:
                    temp += "&"
                else:
                    temp += "#"
                
            temp += "\n"
        return temp + "--------\n"

    fn __copyinit__(out self, other: Self):
        self.pixels = other.pixels
        self.label = other.label

fn readData(count: Int, test_set: String, ptr: UnsafePointer[Image]):
    #print("Reading images in from", test_set)
    var data_filename: String = FILE_TEST_IMAGE
    var label_filename: String = FILE_TEST_LABEL
    if test_set == "train":
        data_filename = FILE_TRAIN_IMAGE
        label_filename = FILE_TRAIN_LABEL
    try:
        var data_file = open(data_filename, "r")
        var label_file = open(label_filename, "r")
    
        _ = data_file.seek(16, os.SEEK_SET)    # is this some header?...
        _ = label_file.seek(8, os.SEEK_SET)  # ...just copying the other work (was 8, wtf?)

        alias buffer_size = IMAGE_SIZE * IMAGE_SIZE
        var image_buffer = UnsafePointer[UInt8].alloc(buffer_size)
        
        for c in range(count): # need to copy over, this is awful. yikes.
            var data_list = data_file.read_bytes(buffer_size)
            
            var temp = label_file.read_bytes(1)
            var data_label: UInt8 = temp[0]

            for i in range(IMAGE_SIZE):
                for j in range(IMAGE_SIZE):
                    idx = i * IMAGE_SIZE + j
                    image_buffer[idx] = data_list[idx]

            var test_image = Image(image_buffer, data_label)
            ptr[c] = test_image

        data_file.close()
        label_file.close()
        image_buffer.free()
    except e:
        print("Error with input binary files")

fn action(x: Scalar[ftype]) -> Scalar[ftype]:
    return x if x > 0 else 0

fn argMax[layout: Layout](output: LayoutTensor[mut = True, ftype, layout, MutableAnyOrigin]) -> Int:
    var largest_value: Scalar[ftype] = FloatLiteral[].negative_infinity
    var pos: Int = 0
    for i in range(layout.size()): # not super flexible depending on rank
        var value = rebind[Scalar[ftype]](output[i])
        if value > largest_value:
            largest_value = value
            pos = i
    return pos

fn softMax[count: Int](input: LayoutTensor[ftype, Layout.row_major(count), MutableAnyOrigin], loss: LayoutTensor[ftype, Layout.row_major(count), MutableAnyOrigin], label: Int) -> None:
    var inner: loss.element_type = 0.0
    for i in range(count):
        var res: input.element_type = 0.0
        for j in range(count):
            #print(input[j], input[i], exp(input[j] - input[i]), end = ";\n")
            res += exp(input[j] - input[i])
        loss[i] = 1.0 / res
        inner -= loss[i] * loss[i]

    inner += loss[label]
    #print("i", inner, end = ", ")
    for i in range(count):
        var temp = 1 if i == label else 0
        loss[i] *= temp - loss[i] - inner

fn loadTarget(features: Feature, errors: Feature, label: Int) -> None:
    softMax(features.output, errors.output, label)

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)
#CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
# actiongrad (return x > 0);
fn convoluteBackward[in_chan: Int,
                     out_chan: Int,
                     feat_size: Int,
                     kernel_size: Int,
                     ](
                             input: LayoutTensor[ftype, Layout.row_major(in_chan, feat_size, feat_size), MutableAnyOrigin],
                             inerror: LayoutTensor[ftype, Layout.row_major(in_chan, feat_size, feat_size), MutableAnyOrigin],
                             outerror: LayoutTensor[ftype, Layout.row_major(out_chan, feat_size - kernel_size + 1, feat_size - kernel_size + 1), MutableAnyOrigin],
                             weight: LayoutTensor[ftype, Layout.row_major(in_chan, out_chan, kernel_size, kernel_size), MutableAnyOrigin],
                             wdeltas: LayoutTensor[ftype, Layout.row_major(in_chan, out_chan, kernel_size, kernel_size), MutableAnyOrigin],
                             bdeltas: LayoutTensor[ftype, Layout.row_major(out_chan), MutableAnyOrigin]):

    alias out_feat_size = feat_size - kernel_size + 1
    # do things
    @parameter
    for x in range(in_chan):
        for y in range(out_chan):
            #outerror[y] inerror[x] weight[x][y] -> input output weight
            var inerror_slice = rebind[LayoutTensor[ftype, Layout.row_major(feat_size, feat_size), MutableAnyOrigin]](inerror.slice[Slice(0, feat_size), Slice(0, feat_size), IndexList[2](1,2)](IndexList[2](x)))

            var weight_slice = rebind[LayoutTensor[ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin]](weight.slice[Slice(0, kernel_size), Slice(0, kernel_size), IndexList[2](2,3)](IndexList[2](x,y))) # check shapes

            var outerror_slice = rebind[LayoutTensor[ftype, Layout.row_major(out_feat_size, out_feat_size), MutableAnyOrigin]](outerror.slice[Slice(0, out_feat_size), Slice(0, out_feat_size), IndexList[2](1,2)](IndexList[2](y)))
            convoluteFull(weight_slice, outerror_slice, inerror_slice )

    @parameter
    for c in range(in_chan): # each element gets "actiongrad"
        for m in range(feat_size):
            for n in range(feat_size):
                inerror[c, m, n] *= 1 if input[c, m, n] > 0 else 0
    
    @parameter
    for c in range(out_chan):
        for i in range(out_feat_size):
            for j in range(out_feat_size):
                bdeltas[c] += outerror[c, i, j]

    for x in range(in_chan):
        for y in range(out_chan):
            #input[x], wd[x][y], outerror[y]
            var input_slice = rebind[LayoutTensor[ftype, Layout.row_major(feat_size, feat_size), MutableAnyOrigin]](input.slice[Slice(0, feat_size), Slice(0, feat_size), IndexList[2](1,2)](IndexList[2](x)))

            var wdeltas_slice = rebind[LayoutTensor[ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin]](wdeltas.slice[Slice(0, kernel_size), Slice(0, kernel_size), IndexList[2](2,3)](IndexList[2](x,y))) # check shapes

            var outerror_slice = rebind[LayoutTensor[ftype, Layout.row_major(out_feat_size, out_feat_size), MutableAnyOrigin]](outerror.slice[Slice(0, out_feat_size), Slice(0, out_feat_size), IndexList[2](1,2)](IndexList[2](y)))
            
            convoluteValid(outerror_slice, input_slice, wdeltas_slice) # input output weight


# CONVOLUTE_VALID(input,output,weight) became (weight, input, output)
fn convoluteValid[feat_size: Int,
                     kernel_size: Int,
                     ](
                        kernel: LayoutTensor[mut = True, ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin],
                        image: LayoutTensor[mut = True, ftype, Layout.row_major(feat_size, feat_size), MutableAnyOrigin],
                        result: LayoutTensor[mut = True, ftype, Layout.row_major(feat_size - kernel_size + 1, feat_size - kernel_size + 1), MutableAnyOrigin]
                     ) -> None:
    for i in range(result.shape[0]()): # each output pixel row
        for j in range(result.shape[1]()): # each output pixel column
            for a in range(kernel.shape[0]()): # for each weight row of a kernel
                for b in range(kernel.shape[1]()): # for each weight col of a kernel
                    result[i, j] +=  image[i + a, j + b] * kernel[a, b]

# CONVOLUTE_FULL(input,output,weight)	why did i reorder to weight, in, out
fn convoluteFull[feat_size: Int,
                     kernel_size: Int,
                     ](
                        kernel: LayoutTensor[ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin],
                        image: LayoutTensor[ftype, Layout.row_major(feat_size - kernel_size + 1, feat_size - kernel_size + 1), MutableAnyOrigin],
                        result: LayoutTensor[ftype, Layout.row_major(feat_size, feat_size), MutableAnyOrigin]
                     ) -> None:
    for i in range(image.shape[0]()): # each input pixel row
        for j in range(image.shape[1]()): # each input pixel column
            for a in range(kernel.shape[0]()): # for each weight row of a kernel
                for b in range(kernel.shape[1]()): # for each weight col of a kernel
                    #print(image[i, j] * kernel[a, b], end = ",, ")
                    result[i + a, j + b] +=  image[i, j] * kernel[a, b]

fn convoluteForward[in_chan: Int,
                     out_chan: Int,
                     feat_size: Int,
                     kernel_size: Int,
                     ](
                        kernels: LayoutTensor[mut = True, ftype, Layout.row_major(in_chan, out_chan, kernel_size, kernel_size)],
                        bias: LayoutTensor[mut = True, ftype, Layout.row_major(out_chan)],
                        image: LayoutTensor[mut = True, ftype, Layout.row_major(in_chan, feat_size, feat_size)],
                        result: LayoutTensor[mut = True, ftype, Layout.row_major(out_chan, feat_size - kernel_size + 1, feat_size - kernel_size + 1)]
                     ) -> None:
    alias out_feat_size = feat_size - kernel_size + 1
    for x in range(kernels.shape[0]()): # number of input channels
        for y in range(kernels.shape[1]()): # number of output channels
            # slicing syntax (gives a 2d for now) = [ Slice(rows wanted), Slice(cols wanted) IndexList[2](dimensions you want) ] (IndexList[2](dim0, dim1) # etc, or can just be a Scalar offset for each dim to use)
            var kern_slice = rebind[LayoutTensor[mut = True, ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin]](kernels.slice[Slice(0, kernel_size), Slice(0, kernel_size), IndexList[2](2,3)](IndexList[2](x,y)))
            
            var image_slice = rebind[LayoutTensor[mut = True, ftype, Layout.row_major(feat_size, feat_size), MutableAnyOrigin]](image.slice[Slice(0, feat_size), Slice(0, feat_size), IndexList[2](1,2)](x)) # might be wrong final arg

            var result_slice = rebind[LayoutTensor[mut = True, ftype, Layout.row_major(out_feat_size, out_feat_size), MutableAnyOrigin]](result.slice[Slice(0, out_feat_size), Slice(0, out_feat_size), IndexList[2](1,2)](y))

            convoluteValid[feat_size, kernel_size](kern_slice, image_slice, result_slice)

            _ = """
            #convoluteValid
            for i in range(result.shape[1]()): # each output pixel row
                for j in range(result.shape[2]()): # each output pixel column
                    for a in range(kernels.shape[2]()): # for each weight row of a kernel
                        for b in range(kernels.shape[2]()): # for each weight col of a kernel
                            result[y, i, j] +=  image[x, i + a, j + b] * kernels[x, y, a, b]
            """
    # activation function (named "action")
    for c in range(result.shape[0]()):
        for i in range(result.shape[1]()):
            for j in range(result.shape[2]()):
                result[c, i, j] += bias[c]
                result[c, i, j] = result[c, i, j] if result[c, i, j] > 0.0 else 0.0 

# SUBSAMP_MAX_BACKWARD(input,inerror,outerror)
# SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
# "out feat size" is from the perspective of the forward pass, sorry if confusing. might want to clear up a lot of names
fn maxPoolBackward[num_channels: Int,
                        in_feat_size: Int,
                        out_feat_size: Int
                      ](
                      input: LayoutTensor[mut = True, ftype, Layout.row_major(num_channels, in_feat_size, in_feat_size), MutableAnyOrigin],
                      inerror: LayoutTensor[mut = True, ftype, Layout.row_major(num_channels, in_feat_size, in_feat_size), MutableAnyOrigin],
                      outerror: LayoutTensor[mut = True, ftype, Layout.row_major(num_channels, out_feat_size, out_feat_size), MutableAnyOrigin]
                      ):
    alias len0 = inerror.shape[1]() // outerror.shape[1]()
    alias len1 = inerror.shape[2]() // outerror.shape[2]()

    for i in range(num_channels):
        for o0 in range(out_feat_size):
            for o1 in range(out_feat_size):
                var x0 = Int(0)
                var x1 = Int(0)
                var ismax: Int

                # branchless approach again
                for l0 in range(len0):
                    for l1 in range(len1):
                        ismax = 1 if input[i, o0 * len0 + l0, o1 * len1 + l1] > input[i, o0 * len0 + x0, o1 * len1 + x1] else 0
                        x0 += ismax * (l0 - x0)
                        x1 += ismax * (l1 - x1)

                inerror[i, o0 * len0 + x0, o1 * len1 + x1] = outerror[i, o0, o1]

# SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
fn maxPoolForward[num_channels: Int,
                        in_feat_size: Int,
                        out_feat_size: Int
                      ](
                      input: LayoutTensor[mut = True, ftype, Layout.row_major(num_channels, in_feat_size, in_feat_size), MutableAnyOrigin],
                      output: LayoutTensor[mut = True, ftype, Layout.row_major(num_channels, out_feat_size, out_feat_size), MutableAnyOrigin]
                      ):
    var lenx = input.shape[1]() // output.shape[1]()
    var leny = input.shape[2]() // output.shape[2]()
    for c in range(output.shape[0]()): # each channel
        for i in range(output.shape[1]()): # feature size
            for j in range(output.shape[2]()): # feature size (should match shape[1]())
                
                var x0: Int = 0
                var y0: Int = 0
                #var ismax: Int

                for x in range(lenx):
                    for y in range(leny):
                        var temp_idx_x = Int(i * lenx + x)
                        var temp_idx_y = Int(j * leny + y)
                        var temp_idx_xx = Int(i * lenx + x0)
                        var temp_idx_yy = Int(j * leny + y0)
                        
                        var ismax = 1 if input[c, temp_idx_x, temp_idx_y] > input[c, temp_idx_xx, temp_idx_yy] else 0
                        x0 += Int(ismax * (x - x0))
                        y0 += Int(ismax * (y - y0))
                
                var temp_idx_xx = Int(i * lenx + x0)
                var temp_idx_yy = Int(j * leny + y0)

                output[c, i, j] = input[c, temp_idx_xx, temp_idx_yy] 

# DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)
# DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
fn matmulBackward[num_chan: Int,
                     feat_size: Int,
                     output_size: Int,
                     ](
                        input: LayoutTensor[mut = True, ftype, Layout.row_major(num_chan, feat_size, feat_size)],
                        inerror: LayoutTensor[mut = True, ftype, Layout.row_major(num_chan, feat_size, feat_size)],
                        outerror: LayoutTensor[mut = True, ftype, Layout.row_major(output_size)],
                        weight: LayoutTensor[mut = True, ftype, Layout.row_major(num_chan * feat_size * feat_size, output_size)],
                        wdeltas: LayoutTensor[mut = True, ftype, Layout.row_major(num_chan * feat_size * feat_size, output_size)],
                        bdeltas: LayoutTensor[mut = True, ftype, Layout.row_major(output_size)]
                     ) -> None:

    alias total_feats = feat_size * feat_size
    for x in range(weight.shape[0]()):
        for y in range(output_size):
            var ie_i = x // (total_feats)
            var rem = x % total_feats
            var ie_j = rem // feat_size
            var ie_k = rem % feat_size
            inerror[ie_i, ie_j, ie_k] += outerror[y] * weight[x, y]

    for i in range(num_chan):
        for j in range(feat_size):
            for k in range(feat_size):
                inerror[i, j, k] *= 1 if input[i, j, k] > 0 else 0

    for i in range(output_size):
        bdeltas[i] += outerror[i]

    for x in range(weight.shape[0]()):
        for y in range(weight.shape[1]()):
            var ie_i = x // (total_feats) # num_chan
            var rem = x % total_feats
            var ie_j = rem // feat_size # feat_size
            var ie_k = rem % feat_size # feat_size
            wdeltas[x, y] += input[ie_i, ie_j, ie_k] * outerror[y]

# 	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
fn matmulForward[num_chan: Int,
                     feat_size: Int,
                     output_size: Int,
                     ](
                        input: LayoutTensor[mut = True, ftype, Layout.row_major(num_chan, feat_size, feat_size)],
                        output: LayoutTensor[mut = True, ftype, Layout.row_major(output_size)],
                        weight: LayoutTensor[mut = True, ftype, Layout.row_major(num_chan * feat_size * feat_size, output_size)],
                        bias: LayoutTensor[mut = True, ftype, Layout.row_major(output_size)]
                     ) -> None:
    # input is m x l, weight is l x n, output is m x n
    # input is (layer5, feat5, feat5), weight is (layer5 * feat5 * feat5, output), output is (output)
    # feat 5 is equal to the value 1
    for x in range(weight.shape[0]()):
        for y in range(weight.shape[1]()):
            for f in range(feat_size):
                output[y] += input[x, f, f] * weight[x, y]
    for i in range(output.shape[0]()):
        output[i] += bias[i]
        output[i] = output[i] if output[i] > 0 else 0

fn loadInput(features: Feature, image: Image):
    var normed = image.toNormalized() # (32, 32) -> (1, 32, 32)
    for i in range(normed.shape[0]()):
        for j in range(normed.shape[1]()):
            features.input[0, i, j] = normed[i, j]

fn forward(lenet: LeNet5, features: Feature):
    convoluteForward(lenet.weight0_1, lenet.bias0_1, features.input, features.layer1)
    # input, l1, lf0, lk

    maxPoolForward(features.layer1, features.layer2)
    # l1 lf1 lf2

    convoluteForward(lenet.weight2_3, lenet.bias2_3, features.layer2, features.layer3)
    #l2 l3 lf2 lk

    maxPoolForward(features.layer3, features.layer4)
    # l3 lf3 lf4

    convoluteForward(lenet.weight4_5, lenet.bias4_5, features.layer4, features.layer5)
    #l4 l5 lf4 lk

    matmulForward(features.layer5, features.output, lenet.weight5_6, lenet.bias5_6)
    #LAYER5, LEA_f5, output

fn backward(lenet: LeNet5, deltas: LeNet5, errors: Feature, features: Feature) -> None:
    matmulBackward(features.layer5, errors.layer5, errors.output, lenet.weight5_6, deltas.weight5_6, deltas.bias5_6)
    #l5, lf5, output

    convoluteBackward(features.layer4, errors.layer4, errors.layer5, lenet.weight4_5, deltas.weight4_5, deltas.bias4_5)
    #l4 l5 lf4 lk

    maxPoolBackward(features.layer3, errors.layer3, errors.layer4)
    #l3 lf3 lf4

    convoluteBackward(features.layer2, errors.layer2, errors.layer3, lenet.weight2_3, deltas.weight2_3, deltas.bias2_3)
    #l2 l3 lf2 lk

    maxPoolBackward(features.layer1, errors.layer1, errors.layer2)
    #l1 lf1 lf2

    convoluteBackward(features.input, errors.input, errors.layer1, lenet.weight0_1, deltas.weight0_1, deltas.bias0_1)
    #input l1 lf0 lk

fn predict(lenet: LeNet5, image: Image) -> Int:
    var feat = Feature()
    loadInput(feat, image)
    forward(lenet, feat)
    return argMax(feat.output)

fn trainBatch(mut lenet: LeNet5, inputs: UnsafePointer[Image], batch_size: Int):
    var buffer = LeNet5()
    var correct = 0

    for i in range(batch_size):
        var feat = Feature()
        var errors = Feature()
        var deltas = LeNet5()
        loadInput(feat, inputs[i])
        forward(lenet, feat)
        var pred = argMax(feat.output)
        var the_label = Int(inputs[i].label)
        if pred == the_label:
            correct += 1
        #print("feat.output", feat.output, "->", pred, "==", the_label)
        loadTarget(feat, errors, the_label)
        #print(errors.output)
        backward(lenet, deltas, errors, feat)
        #var copy = buffer.weight2_3
        buffer.accumulateFromOther(deltas, 1.0)
        _ = """
        for i in range(LAYER2):
            for j in range(LAYER3):
                for k in range(LENGTH_KERNEL):
                    for l in range(LENGTH_KERNEL):
                        if copy[i,j,k,l] != buffer.weight2_3[i,j,k,l]:
                            print("GOOD")
        """
    var k: Scalar[ftype] = Scalar[ftype](ALPHA) / batch_size
    lenet.accumulateFromOther(buffer, k)

    #print(correct, "correct out of", batch_size)

# TODO: UNUSED 
fn train(mut lenet: LeNet5, input: Image, label: Int):
    var feat = Feature()
    var errors = Feature()
    var deltas = LeNet5()

    loadInput(feat, input)
    forward(lenet, feat)
    loadTarget(feat, errors, label)
    backward(lenet, deltas, errors, feat)
    
    lenet.accumulateFromOther(deltas, ALPHA)

fn training(mut lenet: LeNet5, data: UnsafePointer[Image], batch_size: Int, total_size: Int):
    print("Training")
    for i in range(0, total_size, batch_size):
        showProgress(i, total_size)
        var copy = lenet.weight2_3
        trainBatch(lenet, data + i, batch_size)

fn testing(lenet: LeNet5, data: UnsafePointer[Image], total_size: Int) -> Int:
    var correct = 0
    for i in range(total_size):
        var pred = predict(lenet, data[i])
        var actual = Int(data[i].label)
        correct += 1 if pred == actual else 0

    return correct

fn shuffleData(data: UnsafePointer[Image], count: Int, seed: Int = 69):
    if count < 1:
        return
    var rng_state = seed
    #some Claude 4 shit
    for i in range(count - 1, 0, -1):
        rng_state = (rng_state * 1664525 + 1013904223) % 2147483647
        var j = Int(rng_state) % (i + 1)

        var temp = data[i]
        data[i] = data[j]
        data[j] = temp

fn showProgress(progress: Int, total: Int) -> None:
    """
    Not quite working.
    """
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
    #print("hello...", file = stderr)
    var train_data = UnsafePointer[Image].alloc(COUNT_TRAIN)
    var test_data = UnsafePointer[Image].alloc(COUNT_TEST)

    alias tests_to_run = 3
    print(tests_to_run, "tests to run")
    for i in range(tests_to_run):
        seed(i) #random
        #print("reading data in from files")
        readData(COUNT_TRAIN, "train", train_data)
        readData(COUNT_TEST, "test", test_data)

        shuffleData(train_data, COUNT_TRAIN)
        # float32 69 = 97.19%, 420 = 96.98%, 1337 = 97.12%
        # float64 1337 = 97.29% 9001 = 97.25% 69 = 97.04
        # testing f64: 33 97.56

        var lenet = LeNet5()
        lenet.randomizeWeights()
        var batch_size = 300 # could do a number of different batch sizes if we wanted

        #print("begin training")
        training(lenet, train_data, batch_size, COUNT_TRAIN)

        var correct = testing(lenet, test_data, COUNT_TEST)
        print("seed:", i, correct, "/", COUNT_TEST)
    
    _ = """
    var temp_count = 2#Int(COUNT_TRAIN / 10000 * 2)

    print("loading a saved model")
    var model = LeNet5.fromFile[DType.float64]("model_f64.dat")
    var feat: Feature
    correct = 0
    
    alias test_size = COUNT_TEST // 100
    print("testing",  test_size, "images")
    var start_time = perf_counter_ns()

    for c in range(test_size):
        feat = Feature()
        var img = test_data[c]
        loadInput(feat, img)
        forward(model, feat)
        var pred: UInt8 = argMax(feat.output) #[feat.output_layout]
        #print(feat.output)
        #print(pred, end = ", ")
        if pred != img.label:
            #print("\n\tLast Is Incorrect, Actual:")
            #print(String(img))
            correct -= 1
        correct += 1
    var end_time = perf_counter_ns()
    var elapsed_ns = end_time - start_time
    print("test results:", correct, "correct out of ", test_size, "=", correct / test_size * 100, "%")
    print(elapsed_ns, "ns")
    #"""

    # for losers
    train_data.free()
    test_data.free()

