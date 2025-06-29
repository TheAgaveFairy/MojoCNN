from layout import Layout, LayoutTensor, print_layout#, LayoutTensorIter
from layout.layout_tensor import LayoutTensorIter
from math import sqrt
from random import random_float64
from sys.info import sizeof
#import simd
import os


#alias DEBUG = False

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
alias ftype = DType.float32 # model's float type. pixels are uint8 (bytes), non-negotiable

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
        self.weight0_1 = __type_of(self.weight0_1)(w01_storage)
        var w23_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w2_3_layout.size())
        self.weight2_3 = __type_of(self.weight2_3)(w23_storage)
        var w45_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w4_5_layout.size())
        self.weight4_5 = __type_of(self.weight4_5)(w45_storage)
        var w56_storage = UnsafePointer[Scalar[ftype]].alloc(Self.w5_6_layout.size())
        self.weight5_6 = __type_of(self.weight5_6)(w56_storage)

        # BIASES, no more .stack_allocation()
        var b01_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b0_1_layout.size())
        self.bias0_1 = __type_of(self.bias0_1)(b01_storage)
        var b23_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b2_3_layout.size())
        self.bias2_3 = __type_of(self.bias2_3)(b23_storage)
        var b45_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b4_5_layout.size())
        self.bias4_5 = __type_of(self.bias4_5)(b45_storage)
        var b56_storage = UnsafePointer[Scalar[ftype]].alloc(Self.b5_6_layout.size())
        self.bias5_6 = __type_of(self.bias5_6)(b56_storage)

    fn __copyinit__(out self, other: Self):
        self.weight0_1 = other.weight0_1
        self.weight2_3 = other.weight2_3
        self.weight4_5 = other.weight4_5
        self.weight5_6 = other.weight5_6

        self.bias0_1 = other.bias0_1
        self.bias2_3 = other.bias2_3
        self.bias4_5 = other.bias4_5
        self.bias5_6 = other.bias5_6
    
    fn randomizeWeights(self):
        #var iter = LayoutTensorIter[ftype, Layout.row_major(1024)](self.weight0_1)
        for i in range(self.weight0_1.shape[0]()):
            for j in range(self.weight0_1.shape[1]()):
                for k in range(self.weight0_1.shape[2]()):
                    for l in range(self.weight0_1.shape[3]()):
                        self.weight0_1[i,j,k,l] = random_float64(-1.0, 1.0).cast[ftype]()

        for i in range(self.weight2_3.shape[0]()):
            for j in range(self.weight2_3.shape[1]()):
                for k in range(self.weight2_3.shape[2]()):
                    for l in range(self.weight2_3.shape[3]()):
                        self.weight2_3[i,j,k,l] = random_float64(-1.0, 1.0).cast[ftype]()
        
        for i in range(self.weight4_5.shape[0]()):
            for j in range(self.weight4_5.shape[1]()):
                for k in range(self.weight4_5.shape[2]()):
                    for l in range(self.weight4_5.shape[3]()):
                        self.weight4_5[i,j,k,l] = random_float64(-1.0, 1.0).cast[ftype]()

        for i in range(self.weight5_6.shape[0]()):
            for j in range(self.weight5_6.shape[1]()):
                self.weight5_6[i,j] = random_float64(-1.0, 1.0).cast[ftype]()

        for i in range(self.bias0_1.shape[0]()):
            self.bias0_1[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(self.bias2_3.shape[0]()):
            self.bias2_3[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(self.bias4_5.shape[0]()):
            self.bias4_5[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(self.bias5_6.shape[0]()):
            self.bias5_6[i] = random_float64(-1.0, 1.0).cast[ftype]()

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
    for x in range(kernels.shape[0]()): # number of input channels
        for y in range(kernels.shape[1]()): # number of output channels
            for i in range(result.shape[1]()): # each output pixel row
                for j in range(result.shape[2]()): # each output pixel column
                    for a in range(kernels.shape[2]()): # for each weight row of a kernel
                        for b in range(kernels.shape[2]()): # for each weight col of a kernel
                            result[y, i, j] +=  image[x, i + a, j + b] * kernels[x, y, a, b]

    # activation function (named "action")
    for c in range(result.shape[0]()):
        for i in range(result.shape[1]()):
            for j in range(result.shape[2]()):
                result[c, i, j] += bias[c]
                result[c, i, j] = result[c, i, j] if result[c, i, j] > 0.0 else 0.0 

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

fn loadInput(image: Image, features: Feature):
    var normed = image.toNormalized() # (32, 32) -> (1, 32, 32)
    for i in range(normed.shape[0]()):
        for j in range(normed.shape[1]()):
            features.input[0, i, j] = normed[i, j]

fn forward(lenet: LeNet5, features: Feature):
    convoluteForward[INPUT, LAYER1, LENGTH_FEATURE0, LENGTH_KERNEL](
            lenet.weight0_1, lenet.bias0_1, features.input, features.layer1)
    
    maxPoolForward[LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE2](features.layer1, features.layer2)

    convoluteForward[LAYER2, LAYER3, LENGTH_FEATURE2, LENGTH_KERNEL](
            lenet.weight2_3, lenet.bias2_3, features.layer2, features.layer3)
    
    maxPoolForward[LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE4](features.layer3, features.layer4)

    convoluteForward[LAYER4, LAYER5, LENGTH_FEATURE4, LENGTH_KERNEL](
            lenet.weight4_5, lenet.bias4_5, features.layer4, features.layer5)

    matmulForward[LAYER5, LENGTH_FEATURE5, OUTPUT](features.layer5, features.output, lenet.weight5_6, lenet.bias5_6)

def tests():
    alias in_chan = 1
    alias out_chan = 2
    alias image_size = 6
    alias kernel_size = 3
    alias final_size = image_size - kernel_size + 1

    print("test kernel")
    var kernels = LayoutTensor[mut = True, ftype, Layout.row_major(in_chan, out_chan, kernel_size, kernel_size), MutableAnyOrigin].stack_allocation()
    for i in range(kernels.shape[0]()):
        for j in range(kernels.shape[1]()):
            print("chan", i, "->", j)
            for k in range(kernels.shape[2]()):
                for l in range(kernels.shape[3]()):
                    kernels[i, j, k, l] = i + j + k + l
                    print(kernels[i,j,k,l], end = ", ")
                print()
            print()
        print("\n")

    print("test bias")
    var bias = LayoutTensor[mut = True, ftype, Layout.row_major(out_chan), MutableAnyOrigin].stack_allocation()
    for i in range(bias.shape[0]()):
        bias[i] = 0.0005
        print(bias[i], end = ", ")
    print("\n")

    print("test image")
    var image = LayoutTensor[mut = True, ftype, Layout.row_major(in_chan, image_size, image_size), MutableAnyOrigin].stack_allocation()
    for i in range(image.shape[0]()):
        for j in range(image.shape[1]()):
            for k in range(image.shape[2]()):
                image[i, j, k] = j - k
                print(image[i, j, k], end = ", ")
            print("\n")
        print("\n\n")

    var pooled_image = LayoutTensor[mut = True, ftype, Layout.row_major(in_chan, image_size // 2, image_size // 2), MutableAnyOrigin].stack_allocation()

    print("pooling")
    maxPoolForward[1, image_size, image_size // 2](image, pooled_image)
    print(pooled_image)

    var result = LayoutTensor[mut = True, ftype, Layout.row_major(out_chan,final_size,final_size), MutableAnyOrigin].stack_allocation().fill(0.0)
    #CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
    #                         1,5,5           2,4,4              1,2,2,2          2            (RELU)
    for x in range(kernels.shape[0]()): # number of input channels
        for y in range(kernels.shape[1]()): # number of output channels
            for i in range(result.shape[1]()): # each output pixel row
                for j in range(result.shape[2]()): # each output pixel column
                    for a in range(kernels.shape[2]()): # for each weight row of a kernel
                        for b in range(kernels.shape[2]()): # for each weight col of a kernel
                            result[y, i, j] +=  image[x, i + a, j + b] * kernels[x, y, a, b]

    print("result:::")
    for i in range(result.shape[0]()):
        for j in range(result.shape[1]()):
            for k in range(result.shape[2]()):
                print(result[i,j,k], end = ", ")
            print()
        print("\n")

    convoluteForward[in_chan, out_chan, image_size, kernel_size,
                     ](kernels, bias, image, result)
    print("result:::")
    for i in range(result.shape[0]()):
        for j in range(result.shape[1]()):
            for k in range(result.shape[2]()):
                print(result[i,j,k], end = ", ")
            print()
        print("\n")

    print("size of lenet5:", sizeof[LeNet5]())

    # this dont work LMFAO
    var test_tl_copy = LayoutTensor[mut = True, ftype, Layout.row_major(in_chan * image_size * image_size), MutableAnyOrigin].stack_allocation()
    test_tl_copy.copy_from(image)
    print("test image COPIED")
    for i in range(image.shape[0]()):
        for j in range(image.shape[1]()):
            for k in range(image.shape[2]()):
                var temp_idx = Int(i * image.shape[1]() * image.shape[2]() + j * image.shape[2]() + k)
                print(test_tl_copy[temp_idx], end = ", ")
            print("\n")
        print("\n\n")
    

    var test_up = UnsafePointer[Scalar[ftype]].alloc(24)
    for i in range(24):
        test_up[i] = i
    var test_up_tensor = LayoutTensor[ftype, Layout.row_major(2,3,4)](test_up)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                print(test_up_tensor[i,j,k], end = ", ")
            print()
        print()

    print(test_tl_copy.address_space)


    alias temp_layout = Layout.row_major(1,2,1,2)
    var temp_tensor = LayoutTensor[mut = True, ftype, temp_layout, MutableAnyOrigin].stack_allocation()
    var raw_bytes_list: List[Scalar[DType.uint8]] = [0x3f, 0xa0, 0x00, 0x00, 0x40, 0x68, 0x00, 0x00, 0xbf, 0x80, 0x00, 0x00, 0x42, 0x80, 0x00, 0x00] #1.25, 3.625, -1, 64
    var raw_bytes = InlineArray[Scalar[DType.uint8], 16](fill = 0)
    for i in range(16):
        raw_bytes[i] = raw_bytes_list[i]
    LeNet5.bytesToFType[DType.float32, 16, temp_layout](raw_bytes, temp_tensor)
    print(temp_tensor)

def main():
    #print_layout(ImageLayout)

    var train_data = UnsafePointer[Image].alloc(COUNT_TRAIN)
    var test_data = UnsafePointer[Image].alloc(COUNT_TEST)

    var temp_count = 2#Int(COUNT_TRAIN / 10000 * 2)

    readData(COUNT_TRAIN, "train", train_data)
    readData(COUNT_TEST, "test", test_data)
    _ = """
    for i in range(temp_count):
        var train_image = train_data[i]
        print("train sample:\n", String(train_image))
        #print(train_image.toNormalized()[28 * 14])

        var test_image = test_data[i]
        print("test sample:\n", String(test_image))
        #print(test_image.toNormalized()[28 * 14])
    """

    #####################################

    # when it's time to train
    #var model = LeNet5()
    #model.randomizeWeights()

    var model = LeNet5.fromFile[DType.float64]("model_f64.dat")
    var feat: Feature
    var correct = 0
    for c in range(COUNT_TEST):
        feat = Feature()
        var img = test_data[c]
        loadInput(img, feat)
        forward(model, feat)
        var pred: UInt8 = argMax[feat.output_layout](feat.output)
        print("Prediction: ", pred)
        if pred != img.label:
            print(String(img))
            correct -= 1
        correct += 1
    print("test results:", correct, "correct out of ", COUNT_TEST, "=", correct / COUNT_TEST * 100, "%")
    
    _ = """
    var test_image = test_data[0]
    #print(test_data[0].toNormalized().layout, feat.input.layout)
    print(String(test_image))
    loadInput(test_image, feat)
    forward(model, feat)
    print(feat.output)
    print("Prediction: ", argMax[feat.output_layout](feat.output))
    """
    #####################################

    # for losers
    train_data.free()
    test_data.free()


    # tests()
