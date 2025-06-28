import lenet

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

struct LeNet5():
    # LayoutTensor[ftype, Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)]()
    var weight0_1: LayoutTensor[mut = True, ftype, Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL), MutableAnyOrigin]
    var weight2_3: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER2, LAYER3, LENGTH_KERNEL, LENGTH_KERNEL), MutableAnyOrigin]
    var weight4_5: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER4, LAYER5, LENGTH_KERNEL, LENGTH_KERNEL), MutableAnyOrigin]
    var weight5_6: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5, OUTPUT), MutableAnyOrigin]

    var bias0_1: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER1), MutableAnyOrigin]
    var bias2_3: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER3), MutableAnyOrigin]
    var bias4_5: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER5), MutableAnyOrigin]
    var bias5_6: LayoutTensor[mut = True, ftype, Layout.row_major(OUTPUT), MutableAnyOrigin]

    fn __init__(out self):
        self.weight0_1 = __type_of(self.weight0_1).stack_allocation()
        self.weight2_3 = __type_of(self.weight2_3).stack_allocation()
        self.weight4_5 = __type_of(self.weight4_5).stack_allocation()
        self.weight5_6 = __type_of(self.weight5_6).stack_allocation()

        self.bias0_1 = __type_of(self.bias0_1).stack_allocation()
        self.bias2_3 = __type_of(self.bias2_3).stack_allocation()
        self.bias4_5 = __type_of(self.bias4_5).stack_allocation()
        self.bias5_6 = __type_of(self.bias5_6).stack_allocation()
    
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
        if num_elems != tensor.size() or num_elems != len(bytes) / f_sz:
            print("FATAL ERROR CONVERTING BYTES TO TENSOR")
            print(num_elems, tensor.size(), len(bytes))

        if tensor.layout.rank() == 2: # why can't i use "tensor.rank()"?????
            for idx in range(tensor.size()):
                var i = idx // (tensor.shape[1]())
                var j = idx % (tensor.shape[1]())
                var buffer = InlineArray[Scalar[DType.uint8], f_sz](uninitialized = True)
                for bi in range(f_sz):
                    var temp_idx = idx * f_sz + bi
                    buffer[f_sz - 1 - bi] = bytes[temp_idx]
                    #print("into buffer:", buffer[bi])
                var value = SIMD[filetype, 1].from_bytes[](buffer)
                #print("val from raw bytes:", value)
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
                    buffer[f_sz - 1 - bi] = bytes[temp_idx]
                    #print("into buffer:", buffer[bi])
                var value = SIMD[filetype, 1].from_bytes[](buffer)
                #print("val from raw bytes:", value)
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
                    buffer[f_sz - 1 - bi] = bytes[temp_idx]
                var value = SIMD[filetype, 1].from_bytes[](buffer)
                tensor[i,j,k,l] = value.cast[ftype]()

    @staticmethod
    fn fromFile[filetype: DType](filename: String) -> Self:
        alias bytes_per_file_weight = filetype.sizeof()#sizeof[filetype]() must use filetype.sizeof() why?
        var count: Int = 0 # should be 51902 weights in the file
        print("Loading LeNet5 from ", filename, ". filetype number of bytes:", bytes_per_file_weight)
        var model = LeNet5()

        try:
            var model_file = open(filename, "r")
        
            #var bytes = model_file.read_bytes(bytes_to_read)
            #var buffer = InlineArray[Scalar[DType.uint8], bytes_to_read](fill = 0)
            
            alias w01_sz = INPUT * LAYER1 * LENGTH_KERNEL * LENGTH_KERNEL
            print("sizeof w01:", w01_sz)
            alias bytes_to_read = w01_sz * bytes_per_file_weight
            var bytes = model_file.read_bytes(bytes_to_read)
            var buffer = InlineArray[Scalar[DType.uint8], bytes_to_read](uninitialized = True)
            for i in range(bytes_to_read):
                buffer[i] = bytes[i] # could reverse this here, etc
            Self.bytesToFType[filetype, bytes_to_read, model.weight0_1.layout](buffer, model.weight0_1)
            print(model.weight0_1)
            

            print("sizeof w23:", model.weight2_3.size())
            print("sizeof w45:", model.weight4_5.size())
            print("sizeof w56:", model.weight5_6.size())
            
            print("sizeof b01:", model.bias0_1.size())
            print("sizeof b23:", model.bias2_3.size())
            print("sizeof b45:", model.bias4_5.size())
            print("sizeof b56:", model.bias5_6.size())

        except e:
            print("error at reading lenet5 from file", e)
        if count != NUM_WEIGHTS:
            print("ERROR WITH FILE. INCORRECT NUMBER OF WEIGHTS READ")
        return Self()
        

struct Feature():
    # LayoutTensor[ftype, Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)]()
    var input: LayoutTensor[mut = True, ftype, Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0), MutableAnyOrigin]
    var layer1: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1), MutableAnyOrigin]
    var layer2: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2), MutableAnyOrigin]
    var layer3: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3), MutableAnyOrigin]
    var layer4: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4), MutableAnyOrigin]
    var layer5: LayoutTensor[mut = True, ftype, Layout.row_major(LAYER5, LENGTH_FEATURE5, LENGTH_FEATURE5), MutableAnyOrigin]
    var output: LayoutTensor[mut = True, ftype, Layout.row_major(OUTPUT), MutableAnyOrigin]

    fn __init__(out self):
        self.input = __type_of(self.input).stack_allocation().fill(0.0)
        self.layer1 = __type_of(self.layer1).stack_allocation().fill(0.0)
        self.layer2 = __type_of(self.layer2).stack_allocation().fill(0.0)
        self.layer3 = __type_of(self.layer3).stack_allocation().fill(0.0)
        self.layer4 = __type_of(self.layer4).stack_allocation().fill(0.0)
        self.layer5 = __type_of(self.layer5).stack_allocation().fill(0.0)
        self.output = __type_of(self.output).stack_allocation().fill(0.0)
        
alias PixelLayout = Layout.row_major(IMAGE_SIZE, IMAGE_SIZE)
alias PixelStorage = InlineArray[Scalar[DType.uint8], IMAGE_SIZE * IMAGE_SIZE]#(uninitialized = True)
alias PixelTensor = LayoutTensor[mut = True, DType.uint8, PixelLayout] # origin???

alias DataLayout = Layout.row_major(PADDED_SIZE, PADDED_SIZE)
alias DataStorage = InlineArray[Scalar[ftype], PADDED_SIZE * PADDED_SIZE]#
alias DataTensor = LayoutTensor[mut = True, ftype, DataLayout, MutableAnyOrigin] # origin???

struct Image(Stringable, Copyable):
    # we'll store just the 28x28 for now, and write a function to return a padded + normalized version
    #var pixels: InlineArray[Scalar[DType.uint8], IMAGE_SIZE * IMAGE_SIZE]#ImageStorage#ImageTensor
    var pixels: PixelStorage
    var label: UInt8 # [0, 9]

    fn __init__(out self, ptr: UnsafePointer[UInt8], label: UInt8):
        # just stores raw pixel values, normalizing is separate. wasteful and inefficient, i suppose
        #var temp_pixels = InlineArray[Scalar[DType.uint8], IMAGE_SIZE * IMAGE_SIZE](fill = 0)
        var temp_pixels = PixelStorage(uninitialized = True)
        # memcpy probably possible
        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                var idx = r * IMAGE_SIZE + c
                temp_pixels[idx] = ptr[idx]

        #var tensor = PixelTensor(temp_pixels)
        self.pixels = temp_pixels
        self.label = label

    fn toNormalized(self) -> DataStorage:
        # normalizes from 28x28 uint8 to zero-padded 32x32 float32 (or whatever type)
        var storage = DataStorage(fill = 0.0)
        #var tensor = ImageTensor(storage)
        #mut = False gives a terrible terrible compiler warning, please fix
        #var tensor = DataTensor(storage)

        var mean: Float32
        var std: Scalar[ftype]

        var sum: UInt32 = 0
        var std_sum: UInt32 = 0
        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                var idx = r * IMAGE_SIZE + c
                sum += UInt(self.pixels[idx]) # not taking advantage of SIMD possibly?
                std_sum += (UInt32(self.pixels[idx]) * UInt32(self.pixels[idx]))
        
        alias num_elems = IMAGE_SIZE * IMAGE_SIZE
        mean = Float32(sum) / num_elems
        var temp = Float32(std_sum) / num_elems - mean * mean
        std = sqrt(temp)
        #print("sum, std_sum, mean, std", sum, std_sum, mean, std)

        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                var idx = r * IMAGE_SIZE + c
                var new_idx = (r + PADDING) * (IMAGE_SIZE + PADDING * 2) + (c + PADDING)
                # new_idx only adds padding once so it's centered

                #tensor[r + PADDING, c + PADDING] = (Float32(self.pixels[idx]) - mean) / std
                storage[new_idx] = (Float32(self.pixels[idx]) - mean) / std
         
        return storage
        #return ImageTensor(storage) # when do i free

    fn __str__(self) -> String:
        var temp: String = "Label: " + String(self.label) + "\n"
        for row in range(IMAGE_SIZE):
            for col in range(IMAGE_SIZE):
                var idx = row * IMAGE_SIZE + col
                #temp += String(self.pixels[idx]) + ", "
                
                if self.pixels[idx] < 32:
                    temp += " "
                elif self.pixels[idx] < 64:
                    temp += "."
                elif self.pixels[idx] < 96:
                    temp += ","
                elif self.pixels[idx] < 128:
                    temp += "o"
                elif self.pixels[idx] < 160:
                    temp += "x"
                elif self.pixels[idx] < 192:
                    temp += "$"
                elif self.pixels[idx] < 224:
                    temp += "&"
                else:
                    temp += "#"
                
            temp += "\n"
        return temp + "--------\n"

    fn __copyinit__(out self, other: Self):
        self.pixels = other.pixels
        self.label = other.label

fn readData(count: Int, test_set: String, ptr: UnsafePointer[Image]):
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
        
        for i in range(count): # need to copy over, this is awful. yikes.
            var data_list = data_file.read_bytes(buffer_size)
            
            var temp = label_file.read_bytes(1)
            #print("temp: ", temp[1])
            var data_label: UInt8 = temp[0]

            #print("label: ", data_label)
            for i in range(IMAGE_SIZE):
                for j in range(IMAGE_SIZE):
                    idx = i * IMAGE_SIZE + j
                    image_buffer[idx] = data_list[idx]
                    #print(data_list[i * IMAGE_SIZE + j], end = ", ")
                #print()

            var test_image = Image(image_buffer, data_label)
            #print(String(test_image))
            #var testing_delete_me = test_image.toNormalized()
            ptr[i] = test_image
            

        data_file.close()
        label_file.close()
        image_buffer.free()
    except e:
        print("Error with input binary files")

fn action(x: Scalar[ftype]) -> Scalar[ftype]:
    return x if x > 0 else 0

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
                result[c, i, j] = result[c, i, j] if result[c, i, j] > 0.0 else 0.0 + bias[c]

# SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
fn maxPoolForward[num_channels: Int,
                        in_feat_size: Int,
                        out_feat_size: Int
                      ](
                      input: LayoutTensor[mut = True, ftype, Layout.row_major(num_channels, in_feat_size, in_feat_size), MutableAnyOrigin],
                      output: LayoutTensor[mut = True, ftype, Layout.row_major(num_channels, out_feat_size, out_feat_size), MutableAnyOrigin]
                      ):
    var lenx = input.shape[1]() / output.shape[1]()
    var leny = input.shape[2]() / output.shape[2]()
    for c in range(output.shape[0]()): # each channel
        for i in range(output.shape[1]()): # feature size
            for j in range(output.shape[2]()): # feature size (should match shape[1]())
                
                var x0: Int = 0
                var y0: Int = 0
                var ismax: Int

                for x in range(lenx):
                    for y in range(leny):
                        var temp_idx_x = Int(i * lenx + x)
                        var temp_idx_y = Int(i * leny + y)
                        var temp_idx_xx = Int(i * lenx + x0)
                        var temp_idx_yy = Int(i * leny + y0)
                        
                        ismax = 1 if input[c, temp_idx_x, temp_idx_y] > input[c, temp_idx_xx, temp_idx_yy] else 0
                        x0 += Int(ismax * (x - x0))
                        y0 += Int(ismax * (y - y0))
                
                var temp_idx_xx = Int(i * lenx + x0)
                var temp_idx_yy = Int(i * leny + y0)

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
        output[i] = output[i] if output[i] > 0 else 0
        output[i] += bias[i]

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

def main():
    #print_layout(ImageLayout)

    var train_data = UnsafePointer[Image].alloc(COUNT_TRAIN)
    var test_data = UnsafePointer[Image].alloc(COUNT_TEST)

    var temp_count = 2#Int(COUNT_TRAIN / 10000 * 2)

    readData(temp_count, "train", train_data)
    readData(temp_count, "test", test_data)
    for i in range(temp_count):
        var train_image = train_data[i]
        print("train sample:\n", String(train_image))
        #print(train_image.toNormalized()[28 * 14])

        var test_image = test_data[i]
        print("test sample:\n", String(test_image))
        #print(test_image.toNormalized()[28 * 14])

    train_data.free()
    test_data.free()

    #####################################

    var model = LeNet5()
    model.randomizeWeights()
    #print("rank weight0_1 is: ", model.weight0_1.rank)
    var feat = Feature()
    forward(model, feat)


    var model_from_file = LeNet5.fromFile[DType.float64]("model_f64.dat")
    #####################################

    alias in_chan = 1
    alias out_chan = 2
    alias image_size = 5
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
