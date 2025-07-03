from layout import Layout, LayoutTensor, print_layout#, LayoutTensorIter
from layout.layout_tensor import LayoutTensorIter
from math import sqrt
from random import random_float64
from sys.info import sizeof
from utils.index import IndexList
#import simd
from time import perf_counter_ns

def main():
    alias ftype = DType.float32

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

    print(test_up_tensor.layout, "\n\n")

    _ = """
    print("test_up.slice[s(0,3) s(0,4) il[2](0,3)](0)\n",test_up_tensor.slice[Slice(0,3), Slice(0,4), IndexList[2](0,3)](0))
    print()

    print("test_up.slice[s(0,3) s(0,4) il[2](0,2)](1)\n",test_up_tensor.slice[Slice(0,3), Slice(0,4), IndexList[2](0,2)](1))
    print()

    print("test_up.slice[s(0,2) s(0,3) il[2](0,2)](0)\n",test_up_tensor.slice[Slice(0,2), Slice(0,3), IndexList[2](0,2)](0))
    print()
        
    print("test_up.slice[s(0,1) s(0,2) il[2](0,1)](1)\n",test_up_tensor.slice[Slice(0,1), Slice(0,2), IndexList[2](0,1)](1))
    print()

    print("test_up.slice[s(0,1) s(0,2) il[2](0,1)](0)\n",test_up_tensor.slice[Slice(0,1), Slice(0,2), IndexList[2](0,1)](0))
    print()

    for i in range(5):
        print("test_up.slice[s(0,3) s(0,4) il[2](0,1,2,3,4)](i)\n",test_up_tensor.slice[Slice(0,3), Slice(0,4), IndexList[2](0,1,2,3,4)](i))
        print()
    """
    print("test_up.slice[s(0,3) s(0,4) il[2](1,2)](0)\n",test_up_tensor.slice[Slice(0,3), Slice(0,4), IndexList[2](1,2)](0))
    print()

    print("test_up.slice[s(0,3) s(0,4) il[2](1,2)](1)\n",test_up_tensor.slice[Slice(0,3), Slice(0,4), IndexList[2](1,2)](1))
    print()
    

    print("TEST KERNELS")
    var kernels_storage = UnsafePointer[Scalar[ftype]].alloc(48)
    for i in range(48):
        kernels_storage[i] = i
    var kernels = LayoutTensor[ftype, Layout.row_major(2,2,3,4)](kernels_storage)
    print("kernels layout", kernels.layout, "\n\n")
    for in_chan in range(kernels.shape[0]()):
        for out_chan in range(kernels.shape[1]()):
            for row in range(kernels.shape[2]()):
                for col in range(kernels.shape[3]()):
                    print(kernels[in_chan, out_chan, row, col], end = ", ")
                print()
            print()
        print()


    print("kernels.slice[s(0,3) s(0,4) il[3](2,3)](0)\n", kernels.slice[Slice(0,3), Slice(0,4), IndexList[2](2,3)](0))
    print()

    print("kernels.slice[s(0,3) s(0,4) il[2](1,2)](il[2](1,0))\n", kernels.slice[Slice(0,3), Slice(0,4), IndexList[2](2,3)](IndexList[2](1,0)))
    print()

    for in_chan in range(kernels.shape[0]()):
        for out_chan in range(kernels.shape[1]()):
            print("in_chan, out_chan", in_chan, out_chan)
            print(kernels.slice[Slice(0,3), Slice(0,4), IndexList[2](2,3)](IndexList[2](in_chan, out_chan)))

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

    
    print("kernels.slice[(0,ks)(0,ks)[2](0,ks)](0)\n",kernels.slice[Slice(0,kernel_size), Slice(0,kernel_size), IndexList[2](0,kernel_size)](0)) 
    print("kernels.slice[(0,ks)(0,ks)[2](0,ks)](1)\n",kernels.slice[Slice(0,kernel_size), Slice(0,kernel_size), IndexList[2](0,kernel_size)](1)) 

