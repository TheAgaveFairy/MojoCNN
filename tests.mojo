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

fn convoluteFullGPU[
        layout: Layout, kernel_layout: Layout
        ](
        output: LayoutTensor[mut = True, itype, layout],
        input: LayoutTensor[mut = False, itype, layout],
        kernel: LayoutTensor[mut = False, itype, kernel_layout]
        ) -> None:

    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    
    alias img_size = input.shape[0]() # one dim (i.e. 4x4 image = 4)
    alias kernel_size = kernel.shape[0]()
    alias half = kernel_size // 2

    var local_kernel = LayoutTensorBuild[itype]().row_major[kernel_size, kernel_size]().shared().alloc()

    if col < kernel_size and row < kernel_size:
        local_kernel[row, col] = kernel[row, col]

    if row < img_size and col < img_size:
        var result: output.element_type = 0

        @parameter
        for i in range(-half, half + 1):
            @parameter
            for j in range(-half, half + 1):
                var in_row = row + i
                var in_col = col + j

                if in_row >= 0 and in_col >= 0 and in_row < img_size and in_col < img_size:
                    result += input[in_row, in_col] * local_kernel[i + 1, j + 1]

        output[row, col] = result

fn matmulGPU[m: Int, l: Int, n: Int](
        a: LayoutTensor[mut = False, itype, Layout.row_major(m,l), MutableAnyOrigin],
        b: LayoutTensor[mut = False, itype, Layout.row_major(l,n), MutableAnyOrigin],
        c: LayoutTensor[mut = True,  itype, Layout.row_major(m,n), MutableAnyOrigin]
        ) -> None:
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x

    if row < m and col < n:
        var temp: c.element_type = 0
        for k in range(l):
            temp += a[row, k] * b[k, col]
        c[row, col] = temp

fn tensorAddGPU[layout: Layout](
        a: LayoutTensor[mut = False, itype, layout, MutableAnyOrigin],
        b: LayoutTensor[mut = False, itype, layout, MutableAnyOrigin],
        c: LayoutTensor[mut = True,  itype, layout, MutableAnyOrigin]
        ) -> None:
    alias rows = a.shape[0]()
    alias cols = a.shape[1]()
    alias col = block_dim.x * block_idx.x + thread_idx.x
    alias row = block_dim.y * block_idx.y + thread_idx.y

    if col < cols and row < rows:
        c[row, col] = a[row, col] + b[row, col]

fn maxPoolForwardGPU[in_layout: Layout, out_layout: Layout](
        input: LayoutTensor[mut = False, itype, in_layout, MutableAnyOrigin],
        output: LayoutTensor[mut = True, itype, out_layout, MutableAnyOrigin]) -> None:
    
    alias rows = input.shape[0]() // 2
    alias cols = input.shape[1]() // 2
    var col = block_dim.x * block_idx.x + thread_idx.x
    var row = block_dim.y * block_idx.y + thread_idx.y
    
    if row < rows and col < cols:
        var tr = row * 2
        var tc = col * 2

        var temp: output.element_type = max(input[tr, tc], input[tr, tc + 1])
        temp = max(temp, input[tr + 1, tc])
        temp = max(temp, input[tr + 1, tc + 1])
        
        output[row, col] = temp

def main():
    
    # https://www.youtube.com/watch?v=0urE4l4XV98

    alias kernel_size = 3
    alias kernels_layout = Layout.row_major(2,2,kernel_size,kernel_size)
    
    var kernels_storage = UnsafePointer[Scalar[itype]].alloc(48)
    randint[itype](kernels_storage, kernels_layout.size(), -1, 1)

    var kernels = LayoutTensor[mut = True, itype, kernels_layout](kernels_storage)

    print("TEST KERNELS")
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

    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[itype](result_layout.size()).enqueue_fill(0)
        input = ctx.enqueue_create_buffer[itype](img_layout.size()).enqueue_fill(0)
        out_full_buff = ctx.enqueue_create_buffer[itype](img_layout.size()).enqueue_fill(0)
        kern = ctx.enqueue_create_buffer[itype](test_kernel.size()).enqueue_fill(0)
        c_dev = ctx.enqueue_create_buffer[itype](test_kernel.size()).enqueue_fill(0)
        dev_ptr_mp = ctx.enqueue_create_buffer[itype](test_kernel.size()).enqueue_fill(0)

        with input.map_to_host() as inp:
            for i in range(img_layout.size()):
                inp[i] = img.ptr[i]

        with kern.map_to_host() as kr:
            for i in range(test_kernel.layout.size()):
                kr[i] = test_kernel.ptr[i]

        alias BLOCKS_PER_GRID = (1, 1)
        alias THREADS_PER_BLOCK = (out_size, out_size)
        
        dev_out = __type_of(output)(out.unsafe_ptr()) # valid conv output
        dev_out_full = __type_of(img)(out_full_buff.unsafe_ptr()) # full conv output
        dev_img = __type_of(img)(input.unsafe_ptr()) # some test image input
        dev_kern = __type_of(test_kernel)(kern.unsafe_ptr()) # a test kernel
        dev_c = __type_of(test_kernel)(c_dev.unsafe_ptr()) # matmul result
        dev_mp = __type_of(test_kernel)(dev_ptr_mp.unsafe_ptr()) # maxpool result
        
        # VALID
        ctx.enqueue_function[convoluteValidGPU[result_layout, img_layout, test_kernel.layout]](
                dev_out, dev_img, dev_kern,
                grid_dim = BLOCKS_PER_GRID, block_dim = THREADS_PER_BLOCK)
        ctx.synchronize()
        print("Result Valid:")
        with out.map_to_host() as rs:
            for i in range(out_size):
                for j in range(out_size):
                    print(rs[i * out_size + j], end = ", ")
                print()

        # FULL
        ctx.enqueue_function[convoluteFullGPU[img_layout, test_kernel.layout]](
                dev_out_full, dev_img, dev_kern,
                grid_dim = BLOCKS_PER_GRID, block_dim = (img_size, img_size))
        ctx.synchronize()
        print("Result Full:")
        with out_full_buff.map_to_host() as rs:
            for i in range(img_size):
                for j in range(img_size):
                    print(rs[i * img_size + j], end = ", ")
                print()

        # MATMUL
        alias mln = kernel_size
        ctx.enqueue_function[matmulGPU[mln, mln, mln]](
                dev_kern, dev_kern, dev_c,
                grid_dim = BLOCKS_PER_GRID, block_dim = (kernel_size, kernel_size))
        ctx.synchronize()
        print("Result Matmul:")
        with c_dev.map_to_host() as rs:
            for i in range(kernel_size):
                for j in range(kernel_size):
                    print(rs[i * kernel_size + j], end = ", ")
                print()

        # MAXPOOL
        ctx.enqueue_function[maxPoolForwardGPU[img_layout, test_kernel.layout]](
                dev_img, dev_mp,
                grid_dim = BLOCKS_PER_GRID, block_dim = (kernel_size, kernel_size))
        ctx.synchronize()
        print("Result Maxpool of Img:")
        with dev_ptr_mp.map_to_host() as rs:
            for i in range(kernel_size):
                for j in range(kernel_size):
                    print(rs[i * kernel_size + j], end = ", ")
                print()

    # for the losers out there who don't trust their OS
    kernels.ptr.free()
    img.ptr.free()
    # etc LOL

fn testBytesToFloat():
    alias temp_layout = Layout.row_major(1,2,1,2)
    var temp_tensor = LayoutTensor[mut = True, ftype, temp_layout, MutableAnyOrigin].stack_allocation()
    var raw_bytes_list: List[Scalar[DType.uint8]] = [0x3f, 0xa0, 0x00, 0x00, 0x40, 0x68, 0x00, 0x00, 0xbf, 0x80, 0x00, 0x00, 0x42, 0x80, 0x00, 0x00] #1.25, 3.625, -1, 64
    var raw_bytes = InlineArray[Scalar[DType.uint8], 16](fill = 0)
    for i in range(16):
        raw_bytes[i] = raw_bytes_list[i]
    #LeNet5.bytesToFType[DType.float32, 16, temp_layout](raw_bytes, temp_tensor)
    print(temp_tensor)

    
