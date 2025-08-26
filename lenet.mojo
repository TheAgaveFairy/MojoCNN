from layout import Layout, LayoutTensor
from math import sqrt, exp, log
from random import random_float64
from sys.info import sizeof
from sys import stderr, is_big_endian
from utils.index import IndexList
import os
from memory import memcpy
from time import perf_counter_ns

from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, barrier
from layout.tensor_builder import LayoutTensorBuild

from image import Image
from resultlogger import MultiFileLogger, LeNet5Logger
from helpers import showProgress, reLu, reLuGrad

alias LENGTH_KERNEL = 5
alias LENGTH_KERNEL_SQ = LENGTH_KERNEL * LENGTH_KERNEL

alias LENGTH_FEATURE0 = 32
alias LENGTH_FEATURE1 = (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
alias LENGTH_FEATURE2 = (LENGTH_FEATURE1 >> 1)
alias LENGTH_FEATURE3 = (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
alias LENGTH_FEATURE4 = (LENGTH_FEATURE3 >> 1)
alias LENGTH_FEATURE5 = (LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

alias INPUT  =  1
alias LAYER1 =  6
alias LAYER2 =  LAYER1
alias LAYER3 =  16
alias LAYER4 =  LAYER3
alias LAYER5 =  120
alias OUTPUT =  10

alias NUM_WEIGHTS =     51902 # can be calculated but we're just hardcoding for some easier checks at load/save

alias ALPHA = 0.5
alias PADDING = 2

alias IMAGE_SIZE =      28 # as we read it in from the file, its padded to LENGTH_FEATURE0 (a.k.a. PADDED_SIZE)
alias PADDED_SIZE = IMAGE_SIZE + 2 * PADDING # 32 x 32 is what we want eventually # this should equal LENGTH_FEATURE0
alias ftype = DType.float32 # model's internal float type

struct LeNet5(Copyable):
    """
    The LeNet5 model. In the actual LeCun et al implementation, there is some
    notable sparsity in final layers that is not in this version, as well as
    another linear layer of size 84 just before output.

    Unlike my previous C project, these layers are all on the heap instead of
    the stack.
    """
    # WEIGHTS
    alias w01_layout = Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)
    var w01_storage: UnsafePointer[Scalar[ftype]]
    var weight0_1: LayoutTensor[mut = True, ftype, Self.w01_layout, MutableAnyOrigin]
    
    alias w23_layout = Layout.row_major(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL)
    var w23_storage: UnsafePointer[Scalar[ftype]]
    var weight2_3: LayoutTensor[mut = True, ftype, Self.w23_layout, MutableAnyOrigin]
    
    alias w45_layout = Layout.row_major(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL)
    var w45_storage: UnsafePointer[Scalar[ftype]]
    var weight4_5: LayoutTensor[mut = True, ftype, Self.w45_layout, MutableAnyOrigin]
    
    alias w56_layout = Layout.row_major(LAYER5 * LENGTH_FEATURE5 *  LENGTH_FEATURE5, OUTPUT)
    var w56_storage: UnsafePointer[Scalar[ftype]]
    var weight5_6: LayoutTensor[mut = True, ftype, Self.w56_layout, MutableAnyOrigin]

    # BIASES
    alias b01_layout = Layout.row_major(LAYER1)
    var b01_storage: UnsafePointer[Scalar[ftype]]
    var bias0_1: LayoutTensor[mut = True, ftype, Self.b01_layout, MutableAnyOrigin]
    
    alias b23_layout = Layout.row_major(LAYER3)
    var b23_storage: UnsafePointer[Scalar[ftype]]
    var bias2_3: LayoutTensor[mut = True, ftype, Self.b23_layout, MutableAnyOrigin]
    
    alias b45_layout = Layout.row_major(LAYER5)
    var b45_storage: UnsafePointer[Scalar[ftype]]
    var bias4_5: LayoutTensor[mut = True, ftype, Self.b45_layout, MutableAnyOrigin]
    
    alias b56_layout = Layout.row_major(OUTPUT)
    var b56_storage: UnsafePointer[Scalar[ftype]]
    var bias5_6: LayoutTensor[mut = True, ftype, Self.b56_layout, MutableAnyOrigin]

    fn __init__(out self):
        """
        Initialize to all zeros, for training you'll want to randomizeWeights(),
        or for inference, read in from a file. Only biases really need to be set
        to zeroes.
        """
        self.w01_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w01_layout.size())
        self.weight0_1 = __type_of(self.weight0_1)(self.w01_storage).fill(0.0)

        self.w23_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w23_layout.size())
        self.weight2_3 = __type_of(self.weight2_3)(self.w23_storage).fill(0.0)
        
        self.w45_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w45_layout.size())
        self.weight4_5 = __type_of(self.weight4_5)(self.w45_storage).fill(0.0)
        
        self.w56_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w56_layout.size())
        self.weight5_6 = __type_of(self.weight5_6)(self.w56_storage).fill(0.0)

        # BIASES, no more .stack_allocation()
        self.b01_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b01_layout.size())
        self.bias0_1 = __type_of(self.bias0_1)(self.b01_storage).fill(0.0)
        
        self.b23_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b23_layout.size())
        self.bias2_3 = __type_of(self.bias2_3)(self.b23_storage).fill(0.0)
        
        self.b45_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b45_layout.size())
        self.bias4_5 = __type_of(self.bias4_5)(self.b45_storage).fill(0.0)
        
        self.b56_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b56_layout.size())
        self.bias5_6 = __type_of(self.bias5_6)(self.b56_storage).fill(0.0)

    fn __copyinit__(out self, existing: Self):
        # WEIGHTS
        self.w01_storage = __type_of(self.w01_storage).alloc(Self.w01_layout.size())
        memcpy(self.w01_storage, existing.w01_storage, Self.w01_layout.size())
        self.weight0_1 = __type_of(self.weight0_1)(self.w01_storage)

        self.w23_storage = __type_of(self.w23_storage).alloc(Self.w23_layout.size())
        memcpy(self.w23_storage, existing.w23_storage, Self.w23_layout.size())
        self.weight2_3 = __type_of(self.weight2_3)(self.w23_storage)

        self.w45_storage = __type_of(self.w45_storage).alloc(Self.w45_layout.size())
        memcpy(self.w45_storage, existing.w45_storage, Self.w45_layout.size())
        self.weight4_5 = __type_of(self.weight4_5)(self.w45_storage)

        self.w56_storage = __type_of(self.w56_storage).alloc(Self.w56_layout.size())
        memcpy(self.w56_storage, existing.w56_storage, Self.w56_layout.size())
        self.weight5_6 = __type_of(self.weight5_6)(self.w56_storage)

        # BIASES
        self.b01_storage = __type_of(self.b01_storage).alloc(Self.b01_layout.size())
        memcpy(self.b01_storage, existing.b01_storage, Self.b01_layout.size())
        self.bias0_1 = __type_of(self.bias0_1)(self.b01_storage)

        self.b23_storage = __type_of(self.b23_storage).alloc(Self.b23_layout.size())
        memcpy(self.b23_storage, existing.b23_storage, Self.b23_layout.size())
        self.bias2_3 = __type_of(self.bias2_3)(self.b23_storage)

        self.b45_storage = __type_of(self.b45_storage).alloc(Self.b45_layout.size())
        memcpy(self.b45_storage, existing.b45_storage, Self.b45_layout.size())
        self.bias4_5 = __type_of(self.bias4_5)(self.b45_storage)

        self.b56_storage = __type_of(self.b56_storage).alloc(Self.b56_layout.size())
        memcpy(self.b56_storage, existing.b56_storage, Self.b56_layout.size())
        self.bias5_6 = __type_of(self.bias5_6)(self.b56_storage)

    fn __moveinit__(out self, owned existing: Self):
        self.w01_storage = existing.w01_storage
        self.w23_storage = existing.w23_storage
        self.w45_storage = existing.w45_storage
        self.w56_storage = existing.w56_storage

        self.b01_storage = existing.b01_storage
        self.b23_storage = existing.b23_storage
        self.b45_storage = existing.b45_storage
        self.b56_storage = existing.b56_storage

        self.weight0_1 = __type_of(self.weight0_1)(self.w01_storage)
        existing.w01_storage = __type_of(existing.w01_storage)()

        self.weight2_3 = __type_of(self.weight2_3)(self.w23_storage)
        existing.w23_storage = __type_of(existing.w23_storage)()

        self.weight4_5 = __type_of(self.weight4_5)(self.w45_storage)
        existing.w45_storage = __type_of(existing.w45_storage)()

        self.weight5_6 = __type_of(self.weight5_6)(self.w56_storage)
        existing.w56_storage = __type_of(existing.w56_storage)()

        self.bias0_1 = __type_of(self.bias0_1)(self.b01_storage)
        existing.b01_storage = __type_of(existing.b01_storage)()

        self.bias2_3 = __type_of(self.bias2_3)(self.b23_storage)
        existing.b23_storage = __type_of(existing.b23_storage)()

        self.bias4_5 = __type_of(self.bias4_5)(self.b45_storage)
        existing.b45_storage = __type_of(existing.b45_storage)()

        self.bias5_6 = __type_of(self.bias5_6)(self.b56_storage)
        existing.b56_storage = __type_of(existing.b56_storage)()

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
        """
        For taking in errors / deltas during backward pass with learning rate.
        The 'other.layer * lr' doesn't compile, though it seems it should.
        """
        _ = """
        self.weight0_1 += other.weight0_1 * lr
        self.weight2_3 += other.weight2_3 * lr 
        self.weight4_5 += other.weight4_5 * lr
        self.weight5_6 += other.weight5_6 * lr

        self.bias0_1 += other.bias0_1 * lr
        self.bias2_3 += other.bias2_3 * lr 
        self.bias4_5 += other.bias4_5 * lr 
        self.bias5_6 += other.bias5_6 * lr
        """

        for i in range(self.weight0_1.shape[0]()):
            for j in range(self.weight0_1.shape[1]()):
                for k in range(self.weight0_1.shape[2]()):
                    for l in range(self.weight0_1.shape[3]()):
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
        """
        For initializing for training. Biases stay at zeros.
        There might be a better way to do this (SIMD, flatten and @parameter).
        """

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

    @staticmethod
    fn bytesHelper[filetype: DType](buffer: InlineArray[UInt8, filetype.sizeof()]) -> Scalar[filetype]:
        """
        Filetype might be able to be a runtime parameter instead.
        The "from_bytes()" method stopped working for me after an update, so
        the source code was copy / pasted here to get it working.
        Takes in some "bytes" and casts them to the desired type.
        """
        alias f_sz = filetype.sizeof()
        var result: Scalar[filetype]

        @parameter
        if is_big_endian():
            var reversed = __type_of(buffer)(fill = 0)#(uninitialized = True)
            for b in range(f_sz):
                reversed[b] = buffer[f_sz - 1 - b] 
            result = reversed.unsafe_ptr().bitcast[Scalar[filetype]]()[]
        else:
            result = buffer.unsafe_ptr().bitcast[Scalar[filetype]]()[]

        return result

    @staticmethod
    fn bytesToFType[filetype: DType, num_bytes: Int, layout: Layout]
        (bytes: InlineArray[Scalar[DType.uint8], num_bytes],
         tensor: LayoutTensor[mut = True, ftype, layout, MutableAnyOrigin]) -> None:
        """
        Helper function that takes in an array of bytes from a "model.dat" file
        and converts them to the correct datatype and fills the associated layer.
        """
        
        alias f_sz = filetype.sizeof()
        alias num_elems = num_bytes // f_sz

        if num_elems != tensor.size():
            print("FATAL ERROR CONVERTING BYTES TO TENSOR") # TODO: better error
            print("num_elems, tensor.size(), len(bytes):", num_elems, tensor.size(), len(bytes))

        # another way to do this, flatten everything.
        _ = """
        ########
        for idx in range(num_elems):
            var buffer = InlineArray[Scalar[DType.uint8], f_sz](uninitialized = True)
            #var buffer = InlineArray[SIMD[DType.uint8, 1], f_sz](uninitialized = True)
            for byte in range(f_sz):
                buffer[byte] = bytes[idx * f_sz + byte]
            #var value = SIMD[filetype, 1].from_bytes(buffer)
            #var value = Scalar[filetype].from_bytes(buffer)
            #var value = Scalar[DType.float64].from_bytes(buffer)
            #@parameter
            if not is_big_endian():
                for b in range(f_sz // 2):
                    buffer[b], buffer[f_sz - 1 - b] = buffer[f_sz - 1 - b], buffer[b] 
            
            var value = buffer.unsafe_ptr().bitcast[Scalar[filetype]]()[]
            tensor.ptr[idx] = value.cast[ftype]()
        ########
        """
        @parameter
        if layout.rank() == 1: # why can't i use "tensor.rank()"?????
            for idx in range(tensor.size()):
                var i = idx
                
                var buffer = InlineArray[Scalar[DType.uint8], f_sz](fill = 0)#(uninitialized = True)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[bi] = bytes[temp_idx] # f_sz - 1 - bi to reverse
                #var value = SIMD[filetype, 1].from_bytes(buffer)
                var value = Self.bytesHelper[filetype](buffer)
                tensor[i] = value.cast[ftype]()

        elif layout.rank() == 2:
            for idx in range(tensor.size()):
                var i = idx // (tensor.shape[1]())
                var j = idx % (tensor.shape[1]())
                
                var buffer = InlineArray[Scalar[DType.uint8], f_sz](fill = 0)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[bi] = bytes[temp_idx]
                var value = Self.bytesHelper[filetype](buffer)
                tensor[i,j] = value.cast[ftype]()
        
        elif layout.rank() == 3:
            for idx in range(tensor.size()):
                var i = idx // (tensor.shape[1]() * tensor.shape[2]())
                var remainder = idx % (tensor.shape[1]() * tensor.shape[2]())
                var j = remainder // tensor.shape[2]()
                var k = remainder % tensor.shape[2]()

                var buffer = InlineArray[Scalar[DType.uint8], f_sz](fill = 0)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[bi] = bytes[temp_idx]
                var value = Self.bytesHelper[filetype](buffer)
                tensor[i,j,k] = value.cast[ftype]()
                
        elif layout.rank() == 4:
            for idx in range(tensor.size()):
                var i = idx // (tensor.shape[1]() * tensor.shape[2]() * tensor.shape[3]())
                var remainder = idx % (tensor.shape[1]() * tensor.shape[2]() * tensor.shape[3]())
                var j = remainder // (tensor.shape[2]() * tensor.shape[3]())
                remainder = remainder % (tensor.shape[2]() * tensor.shape[3]())
                var k = remainder // tensor.shape[3]()
                var l = remainder % tensor.shape[3]()

                var buffer = InlineArray[Scalar[DType.uint8], f_sz](fill = 0)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[bi] = bytes[temp_idx]
                var value = Self.bytesHelper[filetype](buffer)
                tensor[i,j,k,l] = value.cast[ftype]()

        else:
            print("TENSOR RANK ERROR:", layout.rank(), file=stderr)
            # should be unreachable or cause compiler errors, but we'll
            # be explicit

    @staticmethod
    fn fromFile[filetype: DType](filename: String) -> Self:
        """
        Reads in a "model.dat" file and loads it into a Self.
        Note: Closures can't have parameters, yet.
        """
        alias bytes_per_file_weight = filetype.sizeof()# sizeof[filetype]() won't work, must use filetype.sizeof()

        var model = LeNet5()

        try:
            with open(filename, "r") as model_file:

                # TODO: Check for compiler updates on this closure (needs parameter support)!
                _ = """
                fn helper[layout: Layout](weights: LayoutTensor[mut = True, ftype, layout, MutableAnyOrigin]):
                    alias size_of_layer = layout.size()
                    alias bytes_to_read = size_of_layer * bytes_per_file_weight
                    var bytes: List[UInt8]
                    try:
                        bytes = model_file.read_bytes(bytes_to_read)
                    except ee:
                        print("helper fromFile", ee)
                    var buffer = InlineArray[Scalar[DType.uint8], bytes_to_read](uninitialized = True)
                    for i in range(bytes_to_read):
                        buffer[i] = bytes[i] # memcpy
                    Self.bytesToFType[filetype, bytes_to_read, layout](buffer, weights)

                helper(model.weight0_1) #ETC for each layer
                """

                # WEIGHTS
                alias w01_sz = model.w01_layout.size() # INPUT * LAYER1 * LENGTH_KERNEL * LENGTH_KERNEL
                alias w01_bytes_to_read = w01_sz * bytes_per_file_weight
                var bytes = model_file.read_bytes(w01_bytes_to_read)
                var w01_buffer = InlineArray[Scalar[DType.uint8], w01_bytes_to_read](fill = 0)
                for i in range(w01_bytes_to_read):
                    w01_buffer[i] = bytes[i]
                #Self.bytesToFType[filetype, w01_bytes_to_read, model.w01_layout](w01_buffer, model.weight0_1)
                Self.bytesToFType[filetype](w01_buffer, model.weight0_1)

                alias w23_sz = model.w23_layout.size()
                alias w23_bytes_to_read = w23_sz * bytes_per_file_weight
                bytes = model_file.read_bytes(w23_bytes_to_read)
                var w23_buffer = InlineArray[Scalar[DType.uint8], w23_bytes_to_read](fill = 0)
                for i in range(w23_bytes_to_read):
                    w23_buffer[i] = bytes[i]
                Self.bytesToFType[filetype](w23_buffer, model.weight2_3)

                alias w45_sz = model.w45_layout.size()
                alias w45_bytes_to_read = w45_sz * bytes_per_file_weight
                bytes = model_file.read_bytes(w45_bytes_to_read)
                var w45_buffer = InlineArray[Scalar[DType.uint8], w45_bytes_to_read](fill = 0)
                for i in range(w45_bytes_to_read):
                    w45_buffer[i] = bytes[i]
                Self.bytesToFType[filetype](w45_buffer, model.weight4_5)

                alias w56_sz = model.w56_layout.size()
                alias w56_bytes_to_read = w56_sz * bytes_per_file_weight
                bytes = model_file.read_bytes(w56_bytes_to_read)
                var w56_buffer = InlineArray[Scalar[DType.uint8], w56_bytes_to_read](fill = 0)
                for i in range(w56_bytes_to_read):
                    w56_buffer[i] = bytes[i]
                Self.bytesToFType[filetype](w56_buffer, model.weight5_6)

                # BIASES
                alias b01_sz = model.b01_layout.size()
                alias b01_bytes_to_read = b01_sz * bytes_per_file_weight
                bytes = model_file.read_bytes(b01_bytes_to_read)
                var b01_buffer = InlineArray[Scalar[DType.uint8], b01_bytes_to_read](fill = 0)
                for i in range(b01_bytes_to_read):
                    b01_buffer[i] = bytes[i]
                Self.bytesToFType[filetype](b01_buffer, model.bias0_1)

                alias b23_sz = model.b23_layout.size()
                alias b23_bytes_to_read = b23_sz * bytes_per_file_weight
                bytes = model_file.read_bytes(b23_bytes_to_read)
                var b23_buffer = InlineArray[Scalar[DType.uint8], b23_bytes_to_read](fill = 0)
                for i in range(b23_bytes_to_read):
                    b23_buffer[i] = bytes[i]
                Self.bytesToFType[filetype](b23_buffer, model.bias2_3)

                alias b45_sz = model.b45_layout.size()
                alias b45_bytes_to_read = b45_sz * bytes_per_file_weight
                bytes = model_file.read_bytes(b45_bytes_to_read)
                var b45_buffer = InlineArray[Scalar[DType.uint8], b45_bytes_to_read](fill = 0)
                for i in range(b45_bytes_to_read):
                    b45_buffer[i] = bytes[i]
                Self.bytesToFType[filetype](b45_buffer, model.bias4_5)

                alias b56_sz = model.b56_layout.size()
                alias b56_bytes_to_read = b56_sz * bytes_per_file_weight
                bytes = model_file.read_bytes(b56_bytes_to_read)
                var b56_buffer = InlineArray[Scalar[DType.uint8], b56_bytes_to_read](fill = 0)
                for i in range(b56_bytes_to_read):
                    b56_buffer[i] = bytes[i]
                Self.bytesToFType[filetype](b56_buffer, model.bias5_6)

        except e:
            print("error at reading lenet5 from file", e)
        finally:
            return model

struct Feature():
    """
    These buffers hold intermediate results. Wish it could easily be on the stack
    instead of heap, not sure about how to make that happen.
    """
    alias input_layout = Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0)
    var input_storage: UnsafePointer[Scalar[ftype]]
    var input: LayoutTensor[mut = True, ftype, Feature.input_layout, MutableAnyOrigin]

    alias layer1_layout = Layout.row_major(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1)
    var layer1_storage: UnsafePointer[Scalar[ftype]]
    var layer1: LayoutTensor[mut = True, ftype, Feature.layer1_layout, MutableAnyOrigin]

    alias layer2_layout = Layout.row_major(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2)
    var layer2_storage: UnsafePointer[Scalar[ftype]]
    var layer2: LayoutTensor[mut = True, ftype, Feature.layer2_layout, MutableAnyOrigin]

    alias layer3_layout = Layout.row_major(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3)
    var layer3_storage: UnsafePointer[Scalar[ftype]]
    var layer3: LayoutTensor[mut = True, ftype, Feature.layer3_layout, MutableAnyOrigin]
    
    alias layer4_layout = Layout.row_major(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4)
    var layer4_storage: UnsafePointer[Scalar[ftype]]
    var layer4: LayoutTensor[mut = True, ftype, Feature.layer4_layout, MutableAnyOrigin]
    
    alias layer5_layout = Layout.row_major(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5)
    var layer5_storage: UnsafePointer[Scalar[ftype]]
    var layer5: LayoutTensor[mut = True, ftype, Feature.layer5_layout, MutableAnyOrigin]
    
    alias output_layout = Layout.row_major(OUTPUT)
    var output_storage: UnsafePointer[Scalar[ftype]]
    var output: LayoutTensor[mut = True, ftype, Feature.output_layout, MutableAnyOrigin]

    fn __init__(out self):
        """
        Needs to start as all zeros.
        """
        self.input_storage = UnsafePointer[Scalar[ftype]].alloc(Self.input_layout.size())
        self.input = __type_of(self.input)(self.input_storage).fill(0.0)

        self.layer1_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer1_layout.size())
        self.layer1 = __type_of(self.layer1)(self.layer1_storage).fill(0.0)
        
        self.layer2_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer2_layout.size())
        self.layer2 = __type_of(self.layer2)(self.layer2_storage).fill(0.0)

        self.layer3_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer3_layout.size())
        self.layer3 = __type_of(self.layer3)(self.layer3_storage).fill(0.0)

        self.layer4_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer4_layout.size())
        self.layer4 = __type_of(self.layer4)(self.layer4_storage).fill(0.0)

        self.layer5_storage = UnsafePointer[Scalar[ftype]].alloc(Self.layer5_layout.size())
        self.layer5 = __type_of(self.layer5)(self.layer5_storage).fill(0.0)

        self.output_storage = UnsafePointer[Scalar[ftype]].alloc(Self.output_layout.size())
        self.output = __type_of(self.output)(self.output_storage).fill(0.0)

    fn __del__(owned self):
        self.input_storage.free()
        self.layer1_storage.free()
        self.layer2_storage.free()
        self.layer3_storage.free()
        self.layer4_storage.free()
        self.layer5_storage.free()
        self.output_storage.free()

fn argMax[layout: Layout](output: LayoutTensor[mut = True, ftype, layout, MutableAnyOrigin]) -> Int:
    var largest_value: Scalar[ftype] = FloatLiteral[].negative_infinity
    var pos: Int = 0
    for i in range(layout.size()):
        var value = rebind[Scalar[ftype]](output[i])
        if value > largest_value:
            largest_value = value
            pos = i
    return pos

fn crossEntropyLoss[count: Int](preds: LayoutTensor[ftype, Layout.row_major(count), MutableAnyOrigin], label: Int) -> Float32:
    var max_val: Scalar[ftype] = rebind[Scalar[ftype]](preds[0])
    @parameter
    for i in range(1, count):
        if preds[i] > max_val:
            max_val = rebind[Scalar[ftype]](preds[i])

    var exp_sum: Scalar[ftype] = 0.0
    @parameter
    for i in range(count):
        var temp = rebind[Scalar[ftype]](preds[i] - max_val)
        exp_sum += exp(temp)

    var log_prob: Scalar[ftype] = rebind[Scalar[ftype]]((preds[label] - max_val) - log(exp_sum))
    #var rebound_log_prob = rebind[Scalar[DType.float32]](log_prob)
    return -1.0 * Float32(log_prob)

fn softMax[count: Int](input: LayoutTensor[ftype, Layout.row_major(count), MutableAnyOrigin], loss: LayoutTensor[ftype, Layout.row_major(count), MutableAnyOrigin], label: Int) -> None:
    var inner: loss.element_type = 0.0
    for i in range(count):
        var res: input.element_type = 0.0
        for j in range(count):
            res += exp(input[j] - input[i])
        loss[i] = 1.0 / res
        inner -= loss[i] * loss[i]

    inner += loss[label]
    for i in range(count):
        var temp = 1 if i == label else 0
        loss[i] *= temp - loss[i] - inner

fn loadTarget(features: Feature, errors: Feature, label: Int) -> None:
    softMax(features.output, errors.output, label)

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

    @parameter
    for x in range(in_chan):
        for y in range(out_chan):
            var inerror_slice = rebind[LayoutTensor[ftype, Layout.row_major(feat_size, feat_size), MutableAnyOrigin]](inerror.slice[Slice(0, feat_size), Slice(0, feat_size), IndexList[2](1,2)](IndexList[2](x)))

            var weight_slice = rebind[LayoutTensor[ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin]](weight.slice[Slice(0, kernel_size), Slice(0, kernel_size), IndexList[2](2,3)](IndexList[2](x,y)))

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

            var wdeltas_slice = rebind[LayoutTensor[ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin]](wdeltas.slice[Slice(0, kernel_size), Slice(0, kernel_size), IndexList[2](2,3)](IndexList[2](x,y)))

            var outerror_slice = rebind[LayoutTensor[ftype, Layout.row_major(out_feat_size, out_feat_size), MutableAnyOrigin]](outerror.slice[Slice(0, out_feat_size), Slice(0, out_feat_size), IndexList[2](1,2)](IndexList[2](y)))
            
            convoluteValid(outerror_slice, input_slice, wdeltas_slice)


fn convoluteValid[feat_size: Int,
                     kernel_size: Int,
                     ](
                        kernel: LayoutTensor[mut = True, ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin],
                        image: LayoutTensor[mut = True, ftype, Layout.row_major(feat_size, feat_size), MutableAnyOrigin],
                        result: LayoutTensor[mut = True, ftype, Layout.row_major(feat_size - kernel_size + 1, feat_size - kernel_size + 1), MutableAnyOrigin]
                     ) -> None:
    @parameter
    for i in range(result.shape[0]()): # each output pixel row
        for j in range(result.shape[1]()): # each output pixel column
            for a in range(kernel.shape[0]()): # for each weight row of a kernel
                for b in range(kernel.shape[1]()): # for each weight col of a kernel
                    result[i, j] +=  image[i + a, j + b] * kernel[a, b]

fn convoluteFull[feat_size: Int,
                     kernel_size: Int,
                     ](
                        kernel: LayoutTensor[ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin],
                        image: LayoutTensor[ftype, Layout.row_major(feat_size - kernel_size + 1, feat_size - kernel_size + 1), MutableAnyOrigin],
                        result: LayoutTensor[ftype, Layout.row_major(feat_size, feat_size), MutableAnyOrigin]
                     ) -> None:
    @parameter
    for i in range(image.shape[0]()): # each input pixel row
        for j in range(image.shape[1]()): # each input pixel column
            for a in range(kernel.shape[0]()): # for each weight row of a kernel
                for b in range(kernel.shape[1]()): # for each weight col of a kernel
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
    
    @parameter
    for x in range(kernels.shape[0]()): # number of input channels
        for y in range(kernels.shape[1]()): # number of output channels
            # slicing syntax (gives a 2d for now) = [ Slice(rows wanted), Slice(cols wanted) IndexList[2](dimensions you want) ] (IndexList[2](dim0, dim1) # etc, or can just be a Scalar offset for each dim to use)
            var kern_slice = rebind[LayoutTensor[mut = True, ftype, Layout.row_major(kernel_size, kernel_size), MutableAnyOrigin]](kernels.slice[Slice(0, kernel_size), Slice(0, kernel_size), IndexList[2](2,3)](IndexList[2](x,y)))
            
            var image_slice = rebind[LayoutTensor[mut = True, ftype, Layout.row_major(feat_size, feat_size), MutableAnyOrigin]](image.slice[Slice(0, feat_size), Slice(0, feat_size), IndexList[2](1,2)](x)) # might be wrong final arg

            var result_slice = rebind[LayoutTensor[mut = True, ftype, Layout.row_major(out_feat_size, out_feat_size), MutableAnyOrigin]](result.slice[Slice(0, out_feat_size), Slice(0, out_feat_size), IndexList[2](1,2)](y))

            convoluteValid[feat_size, kernel_size](kern_slice, image_slice, result_slice)

    # activation function (named "action")
    @parameter
    for c in range(result.shape[0]()):
        for i in range(result.shape[1]()):
            for j in range(result.shape[2]()):
                result[c, i, j] += bias[c]
                result[c, i, j] = result[c, i, j] if result[c, i, j] > 0.0 else 0.0 

# "out feat size" is from the perspective of the forward pass... I might want to clear up names
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

    @parameter
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

fn maxPoolForward[num_channels: Int,
                        in_feat_size: Int,
                        out_feat_size: Int
                      ](
                      input: LayoutTensor[mut = True, ftype, Layout.row_major(num_channels, in_feat_size, in_feat_size), MutableAnyOrigin],
                      output: LayoutTensor[mut = True, ftype, Layout.row_major(num_channels, out_feat_size, out_feat_size), MutableAnyOrigin]
                      ):
    var lenx = input.shape[1]() // output.shape[1]()
    var leny = input.shape[2]() // output.shape[2]()

    @parameter
    for c in range(output.shape[0]()): # each channel
        for i in range(output.shape[1]()): # feature size
            for j in range(output.shape[2]()): # feature size (should match shape[1]())
                
                var x0: Int = 0
                var y0: Int = 0

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

    @parameter
    for x in range(weight.shape[0]()):
        for y in range(output_size):
            var ie_i = x // (total_feats)
            var rem = x % total_feats
            var ie_j = rem // feat_size
            var ie_k = rem % feat_size
            inerror[ie_i, ie_j, ie_k] += outerror[y] * weight[x, y]

    @parameter
    for i in range(num_chan):
        for j in range(feat_size):
            for k in range(feat_size):
                inerror[i, j, k] *= 1 if input[i, j, k] > 0 else 0

    @parameter
    for i in range(output_size):
        bdeltas[i] += outerror[i]

    @parameter
    for x in range(weight.shape[0]()):
        for y in range(weight.shape[1]()):
            var ie_i = x // (total_feats) # num_chan
            var rem = x % total_feats
            var ie_j = rem // feat_size # feat_size
            var ie_k = rem % feat_size # feat_size
            wdeltas[x, y] += input[ie_i, ie_j, ie_k] * outerror[y]

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
    # feature_length5 is equal to the value 1
    @parameter
    for x in range(weight.shape[0]()):
        for y in range(weight.shape[1]()):
            for f in range(feat_size):
                output[y] += input[x, f, f] * weight[x, y]
    
    @parameter
    for i in range(output.shape[0]()):
        output[i] += bias[i]
        output[i] = output[i] if output[i] > 0 else 0

fn loadInput(features: Feature, image: Image):
    """
    Reminder: Images must go in as a normalized format. Mild reshaping
    also happens.
    """
    var normed = image.toNormalized() # (32, 32) -> (1, 32, 32)
    for i in range(normed.shape[0]()): # PADDED_SIZE
        for j in range(normed.shape[1]()): # PADDED_SIZE
            features.input[0, i, j] = normed[i, j]
    
    normed.ptr.free()

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
    # TODO: Probably could be a method of LeNet5.
    var feat = Feature()
    loadInput(feat, image)
    forward(lenet, feat)
    return argMax(feat.output)

fn trainBatch(mut model: LeNet5, inputs: UnsafePointer[Image], batch_size: Int) -> Tuple[UInt, Float32]:
    # TODO: Probably could be a method of LeNet5. "correct" ultimately unused
    var buffer = LeNet5()
    var correct = 0
    var total_loss: Float32 = 0.0

    for i in range(batch_size):
        var feat = Feature()
        var errors = Feature()
        var deltas = LeNet5()
        loadInput(feat, inputs[i])
        forward(model, feat)
        var pred = argMax(feat.output)
        var the_label = Int(inputs[i].label)
        if pred == the_label:
            correct += 1
        
        var loss = crossEntropyLoss(feat.output, the_label)
        total_loss += loss
        loadTarget(feat, errors, the_label)
        backward(model, deltas, errors, feat)
        buffer.accumulateFromOther(deltas, 1.0)

    var k: Scalar[ftype] = Scalar[ftype](ALPHA) / batch_size
    model.accumulateFromOther(buffer, k)

    var avg_loss = total_loss / batch_size

    return Tuple[UInt, Float32](correct, avg_loss)

fn training[T: LeNet5Logger](mut model: LeNet5, data: UnsafePointer[Image], batch_size: Int, total_size: Int, mut logger: T):
    #print("Training")
    for i in range(0, total_size, batch_size):
        showProgress(i, total_size)
        var start_time = perf_counter_ns()
        var results_tuple = trainBatch(model, data + i, batch_size)
        var correct = results_tuple[0]
        var avg_loss = results_tuple[1]
        var end_time = perf_counter_ns()
        var elapsed = end_time - start_time
        try:
            logger.logTrainingEpoch("CPU", i, elapsed, correct, total_size, avg_loss, ALPHA, ftype)
        except e:
            print("logging error during CPU training:", e)
        # LOSS, LR

fn training(mut model: LeNet5, data: UnsafePointer[Image], batch_size: Int, total_size: Int):
    #print("Training")
    for i in range(0, total_size, batch_size):
        showProgress(i, total_size)
        _ = trainBatch(model, data + i, batch_size)

fn testing(model: LeNet5, data: UnsafePointer[Image], total_size: Int) -> Int:
    var correct = 0
    for i in range(total_size):
        var pred = predict(model, data[i])
        var actual = Int(data[i].label)
        correct += 1 if pred == actual else 0

    return correct


