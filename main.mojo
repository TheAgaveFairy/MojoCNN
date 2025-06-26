#import lenet

from layout import Layout, LayoutTensor, print_layout
from math import sqrt
from random import random_float64
import os

#alias DEBUG = False

alias FILE_TRAIN_IMAGE =    "train-images-idx3-ubyte"
alias FILE_TRAIN_LABEL =    "train-labels-idx1-ubyte"
alias FILE_TEST_IMAGE =     "t10k-images-idx3-ubyte"
alias FILE_TEST_LABEL =     "t10k-labels-idx1-ubyte"
alias LENET_FILE =          "model.dat"
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
    var weight0_1: InlineArray[Scalar[ftype], INPUT * LAYER1 * LENGTH_KERNEL_SQ]
    var weight2_3: InlineArray[Scalar[ftype], LAYER2 * LAYER3 * LENGTH_KERNEL_SQ]
    var weight4_5: InlineArray[Scalar[ftype], LAYER4 * LAYER5 * LENGTH_KERNEL_SQ]
    var weight5_6: InlineArray[Scalar[ftype], LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5 * OUTPUT]

    var bias0_1: InlineArray[Scalar[ftype], LAYER1]
    var bias2_3: InlineArray[Scalar[ftype], LAYER3]
    var bias4_5: InlineArray[Scalar[ftype], LAYER5]
    var bias5_6: InlineArray[Scalar[ftype], OUTPUT]

    fn __init__(out self):
        self.weight0_1 = InlineArray[Scalar[ftype], INPUT * LAYER1 * LENGTH_KERNEL_SQ](uninitialized = True)  
        self.weight2_3 = InlineArray[Scalar[ftype], LAYER2 * LAYER3 * LENGTH_KERNEL_SQ](uninitialized = True)  
        self.weight4_5 = InlineArray[Scalar[ftype], LAYER4 * LAYER5 * LENGTH_KERNEL_SQ](uninitialized = True)  
        self.weight5_6 = InlineArray[Scalar[ftype], LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5 * OUTPUT](uninitialized = True)  
        #self.weight0_1 = LayoutTensor[mut = True, ftype, Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)](w0_1)

        self.bias0_1 = InlineArray[Scalar[ftype], LAYER1](uninitialized = True)
        self.bias2_3 = InlineArray[Scalar[ftype], LAYER3](uninitialized = True)
        self.bias4_5 = InlineArray[Scalar[ftype], LAYER5](uninitialized = True)
        self.bias5_6 = InlineArray[Scalar[ftype], OUTPUT](uninitialized = True)
    
    fn randomizeWeights(self):
        var model = LeNet5()
        for i in range(len(self.weight0_1)):
            model.weight0_1[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(len(self.weight2_3)):
            model.weight2_3[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(len(self.weight4_5)):
            model.weight4_5[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(len(self.weight5_6)):
            model.weight5_6[i] = random_float64(-1.0, 1.0).cast[ftype]()

        for i in range(len(self.bias0_1)):
            model.bias0_1[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(len(self.bias2_3)):
            model.bias2_3[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(len(self.bias4_5)):
            model.bias4_5[i] = random_float64(-1.0, 1.0).cast[ftype]()
        for i in range(len(self.bias5_6)):
            model.bias5_6[i] = random_float64(-1.0, 1.0).cast[ftype]()

struct Feature():
    # LayoutTensor[ftype, Layout.row_major(INPUT, LAYER1, LENGTH_KERNEL, LENGTH_KERNEL)]()
    var input: InlineArray[Scalar[ftype], INPUT * LENGTH_FEATURE0 * LENGTH_FEATURE0]
    var layer1: InlineArray[Scalar[ftype], LAYER1 * LENGTH_FEATURE1 * LENGTH_FEATURE1]
    var layer2: InlineArray[Scalar[ftype], LAYER2 * LENGTH_FEATURE2 * LENGTH_FEATURE2]
    var layer3: InlineArray[Scalar[ftype], LAYER3 * LENGTH_FEATURE3 * LENGTH_FEATURE3]
    var layer4: InlineArray[Scalar[ftype], LAYER4 * LENGTH_FEATURE4 * LENGTH_FEATURE4]
    var layer5: InlineArray[Scalar[ftype], LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5]
    var output: InlineArray[Scalar[ftype], OUTPUT]

    fn __init__(out self):
        self.input  = InlineArray[Scalar[ftype], INPUT * LENGTH_FEATURE0 * LENGTH_FEATURE0](uninitialized = True)
        self.layer1 = InlineArray[Scalar[ftype], LAYER1 * LENGTH_FEATURE1 * LENGTH_FEATURE1](uninitialized = True)
        self.layer2 = InlineArray[Scalar[ftype], LAYER2 * LENGTH_FEATURE2 * LENGTH_FEATURE2](uninitialized = True)
        self.layer3 = InlineArray[Scalar[ftype], LAYER3 * LENGTH_FEATURE3 * LENGTH_FEATURE3](uninitialized = True)
        self.layer4 = InlineArray[Scalar[ftype], LAYER4 * LENGTH_FEATURE4 * LENGTH_FEATURE4](uninitialized = True)
        self.layer5 = InlineArray[Scalar[ftype], LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5](uninitialized = True)
        self.output = InlineArray[Scalar[ftype], OUTPUT](uninitialized = True)
        
struct Model():
    var w0_1: LayoutTensor[mut = True, ftype, Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0), MutableAnyOrigin]

    fn __init__(out self):
        self.w0_1 = LayoutTensor[mut = True, ftype, Layout.row_major(INPUT, LENGTH_FEATURE0, LENGTH_FEATURE0), MutableAnyOrigin].stack_allocation()

alias PixelLayout = Layout.row_major(IMAGE_SIZE, IMAGE_SIZE)
alias PixelStorage = InlineArray[Scalar[DType.uint8], IMAGE_SIZE * IMAGE_SIZE]#(uninitialized = True)
alias PixelTensor = LayoutTensor[mut = True, DType.uint8, PixelLayout] # origin???

alias DataLayout = Layout.row_major(PADDED_SIZE, PADDED_SIZE)
alias DataStorage = InlineArray[Scalar[ftype], PADDED_SIZE * PADDED_SIZE]#
alias DataTensor = LayoutTensor[mut = True, ftype, DataLayout] # origin???

struct Image(Stringable, Copyable):
    # we'll store just the 28x28 for now, and write a function to return a padded + normalized version
    #var pixels: InlineArray[Scalar[DType.uint8], IMAGE_SIZE * IMAGE_SIZE]#ImageStorage#ImageTensor
    var pixels: PixelStorage
    var label: UInt8 # [0, 9]

    fn __init__(out self, ptr: UnsafePointer[UInt8], label: UInt8):
        # just stores raw pixel values, normalizing is separate. wasteful and inefficient, i suppose
        #var temp_pixels = InlineArray[Scalar[DType.uint8], IMAGE_SIZE * IMAGE_SIZE](fill = 0)
        var temp_pixels = PixelStorage(fill = 0)
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

fn convoluteValid(input: InlineArray, output: InlineArray):
    pass
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
        print(train_image.toNormalized()[28 * 14])

        var test_image = test_data[i]
        print("test sample:\n", String(test_image))
        print(test_image.toNormalized()[28 * 14])

    train_data.free()
    test_data.free()

    var model = LeNet5()
    model.randomizeWeights()
    var feat = Feature()

    var storage = InlineArray[Scalar[ftype], 24](fill = 0.0)
    var tensor = LayoutTensor[ftype, Layout.row_major(2,3,4)](storage)
    print("Tensor created of 24 elems with row major shape (2,3,4):\n\tshape[0]() = ", tensor.shape[0](),\
            "\n\tshape[1]() = ", tensor.shape[1](),\
            "\n\tshape[2]() = ", tensor.shape[2]())
    

    var testing = LayoutTensor[mut = True, ftype, Layout.row_major(3,6,9), MutableAnyOrigin].stack_allocation()
    print("HERE:\n", testing.shape[1]())

    var model_TEST = Model()
