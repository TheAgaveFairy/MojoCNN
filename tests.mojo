from layout import Layout, LayoutTensor, print_layout#, LayoutTensorIter
from layout.layout_tensor import LayoutTensorIter
from math import sqrt
from random import random_float64, randint
from sys.info import sizeof
from utils.index import IndexList
#import simd
from time import perf_counter_ns


from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from memory import stack_allocation, memset_zero
from gpu.memory import AddressSpace
from layout.tensor_builder import LayoutTensorBuild

alias ftype = DType.float32
alias itype = DType.int32

fn convoluteValidGPU[
        out_layout: Layout, in_layout: Layout, kernel_layout: Layout
](
    output: LayoutTensor[mut = True, itype, out_layout],
    img: LayoutTensor[mut = False, itype, in_layout],
    kernel: LayoutTensor[mut = False, itype, kernel_layout],
) -> None:
    # global indices
    gx = block_dim.x * block_idx.x + thread_idx.x # col
    gy = block_dim.y * block_idx.y + thread_idx.y # row
    # local indices
    #lx = thread_idx.x
    #ly = thread_idx.y

    # tb[dtype]().row_major[TPB]().shared().alloc()

    # assuming square inputs
    alias feat_size = img.shape[0]()
    alias kernel_size = kernel.shape[0]()
    alias out_size = feat_size - kernel_size + 1

    var local_kernel = LayoutTensorBuild[itype]().row_major[kernel_size, kernel_size]().shared().alloc()

    if gx < kernel_size and gy < kernel_size:
        local_kernel[gy, gx] = kernel[gy, gx]

    if gx < out_size and gy < out_size:
        var result: output.element_type = 0

        # KERNEL_SIZE dims
        @parameter
        for i in range(kernel_size):
            @parameter
            for j in range(kernel_size):
                var in_row = gy + i
                var in_col = gx + j

                result += img[in_row, in_col] * local_kernel[i, j]

        output[gy, gx] = result

def main():
    print("TEST KERNELS")
    
    #var ctx = DeviceContext()
    # https://www.youtube.com/watch?v=0urE4l4XV98

    alias kernel_size = 3
    alias kernels_layout = Layout.row_major(2,2,kernel_size,kernel_size)
    
    var kernels_storage = UnsafePointer[Scalar[itype]].alloc(48)
    randint[itype](kernels_storage, kernels_layout.size(), -5, 5)

    var kernels = LayoutTensor[mut = True, itype, kernels_layout](kernels_storage)

    for in_chan in range(kernels.shape[0]()):
        for out_chan in range(kernels.shape[1]()):
            print("in_chan, out_chan", in_chan, out_chan)
            print(kernels.slice[Slice(0,3), Slice(0,3), IndexList[2](2,3)](IndexList[2](in_chan, out_chan)))

    # grab kernel at (0, 1)
    var test_kernel = kernels.slice[Slice(0,kernel_size), Slice(0,kernel_size), IndexList[2](2,3)](IndexList[2](0,1))

    alias img_size = 6
    alias img_layout = Layout.row_major(img_size, img_size)
    var img_storage = UnsafePointer[Scalar[itype]].alloc(img_layout.size())
    randint[itype](img_storage, img_layout.size(), -5, 5)
    var img = LayoutTensor[mut = True, itype, img_layout, MutableAnyOrigin](img_storage)
    print("Test Image\n", img)

    alias out_size = img_size - kernel_size + 1
    alias result_layout = Layout.row_major(out_size, out_size)
    var res_storage = UnsafePointer[Scalar[itype]].alloc(result_layout.size())
    memset_zero(res_storage, result_layout.size())
    var output = LayoutTensor[mut = True, itype, result_layout](res_storage)

    #convoluteValidGPU(o i k)
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[itype](result_layout.size()).enqueue_fill(0)
        input = ctx.enqueue_create_buffer[itype](img_layout.size()).enqueue_fill(0)
        kern = ctx.enqueue_create_buffer[itype](test_kernel.size()).enqueue_fill(0)

        with input.map_to_host() as inp:
            for i in range(img_layout.size()):
                inp[i] = img.ptr[i]

        with kern.map_to_host() as kr:
            for i in range(test_kernel.layout.size()):
                kr[i] = test_kernel.ptr[i]

        alias BLOCKS_PER_GRID = (1, 1)
        alias THREADS_PER_BLOCK = (out_size, out_size)
        
        dev_out = __type_of(output)(out.unsafe_ptr())
        dev_img = __type_of(img)(input.unsafe_ptr())
        dev_kern = __type_of(test_kernel)(kern.unsafe_ptr())
        
        ctx.enqueue_function[convoluteValidGPU[result_layout, img_layout, test_kernel.layout]](
                dev_out, dev_img, dev_kern,
                grid_dim = BLOCKS_PER_GRID, block_dim = THREADS_PER_BLOCK)

        ctx.synchronize()
        print("Result:")
        with out.map_to_host() as rs:
            for i in range(out_size):
                for j in range(out_size):
                    print(rs[i * out_size + j], end = ", ")
                print()

    # for the losers out there who don't trust their OS
    kernels.ptr.free()
    img.ptr.free()

fn testBytesToFloat():
    alias temp_layout = Layout.row_major(1,2,1,2)
    var temp_tensor = LayoutTensor[mut = True, ftype, temp_layout, MutableAnyOrigin].stack_allocation()
    var raw_bytes_list: List[Scalar[DType.uint8]] = [0x3f, 0xa0, 0x00, 0x00, 0x40, 0x68, 0x00, 0x00, 0xbf, 0x80, 0x00, 0x00, 0x42, 0x80, 0x00, 0x00] #1.25, 3.625, -1, 64
    var raw_bytes = InlineArray[Scalar[DType.uint8], 16](fill = 0)
    for i in range(16):
        raw_bytes[i] = raw_bytes_list[i]
    #LeNet5.bytesToFType[DType.float32, 16, temp_layout](raw_bytes, temp_tensor)
    print(temp_tensor)

    
