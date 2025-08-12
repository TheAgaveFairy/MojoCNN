from layout import Layout, LayoutTensor, print_layout#, LayoutTensorIter
from layout.layout_tensor import LayoutTensorIter
from math import sqrt, ceil, log2
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
from gpu.warp import shuffle_down

alias ftype = DType.float32
alias itype = DType.int32
#alias ftype = itype

fn matmulGPU[m: Int, l: Int, n: Int, action: fn(Scalar[itype]) -> Scalar[itype]](
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
        c[row, col] = action(rebind[Scalar[itype]](temp))

#####################
alias in_chans = 5
alias out_chans = 2

alias layer5_layout = Layout.row_major(in_chans) 
alias weights_layout = Layout.row_major(in_chans, out_chans)
alias output_layout = Layout.row_major(out_chans)
fn matMulFusedKernel[action: fn(Scalar[itype]) -> Scalar[itype]](
        in_layer: LayoutTensor[mut = True, itype, layer5_layout, MutableAnyOrigin],
        weights: LayoutTensor[mut = True, itype, weights_layout, MutableAnyOrigin],
        output: LayoutTensor[mut = True, itype, output_layout, MutableAnyOrigin]) -> None:
    """
    Enough threads per block to do one output channel at a time as a reduction,
    so make it a power of two.
    Grid Dim = batch_size
    Block Dim = 1 << ceil(log2(in_chans)).
    """
    var thread = thread_idx.x
    alias reduction_size = 1 << Int(ceil(log2(Float64(in_chans)))) # 128 when LAYER5 is 120
    
    #var reduction_buffer = InlineArray[Scalar[itype], reduction_size](fill = 0)
    var reduction_buffer = LayoutTensor[mut = True, itype, Layout.row_major(reduction_size), MutableAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()

    #barrier()
    for oc in range(out_chans):
        if thread < in_chans:
            reduction_buffer[thread] = rebind[Scalar[itype]](weights[thread, oc]) * rebind[Scalar[itype]](in_layer[thread])
        else:
            reduction_buffer[thread] = 0
        barrier()
        #var testingshit = shuffle_down(thread, 1)
        print(thread, reduction_buffer[thread], reduction_buffer[2])

        var i = 1
        while i < reduction_size // 2:
            var temp = reduction_buffer[thread]
            if thread % (2 * i) == 0:
                print("treadno, stride, [thread], [thread + stride]:", thread, i, "->", reduction_buffer[thread], "+", reduction_buffer[thread + i])
                reduction_buffer[thread] += reduction_buffer[thread + i]
            barrier()
            i *= 2
        print("\t", thread, reduction_buffer[thread])

        if thread == 0:
            var temp = rebind[Scalar[itype]](reduction_buffer[0] + 10) # 10.0 for a bias
            output[oc] = action(temp)

fn reLu(x: Scalar[itype]) -> Scalar[itype]:
    return x
    #return x if x > 1 else 0

def main():
    
    # https://www.youtube.com/watch?v=0urE4l4XV98

    with DeviceContext() as ctx:

        var layer5_devstor = ctx.enqueue_create_buffer[itype](layer5_layout.size()).enqueue_fill(0)
        var weights_devstor = ctx.enqueue_create_buffer[itype](weights_layout.size()).enqueue_fill(0)
        var output_devstor = ctx.enqueue_create_buffer[itype](output_layout.size()).enqueue_fill(0)

        with layer5_devstor.map_to_host() as inp:
            for i in range(layer5_layout.size()):
                inp[i] = i

        with weights_devstor.map_to_host() as kr:
            for i in range(weights_layout.size()):
                var row = i // out_chans
                var col = i % out_chans
                kr[i] = 10 * row

        var layer5_dev = LayoutTensor[mut = True, itype, layer5_layout, MutableAnyOrigin](layer5_devstor) # valid conv output
        var weights_dev = LayoutTensor[mut = True, itype, weights_layout, MutableAnyOrigin](weights_devstor) # full conv output
        var output_dev = LayoutTensor[mut = True, itype, output_layout, MutableAnyOrigin](output_devstor) # some test image input
        
        alias BLOCKS_PER_GRID = (1,)
        alias reduction_size = 1 << Int(ceil(log2(Float64(in_chans)))) # 128
        alias THREADS_PER_BLOCK = reduction_size #1 << Int(ceil(log2(Float64(in_chans))))
        ctx.enqueue_function[matMulFusedKernel[reLu]](
                layer5_dev, weights_dev, output_dev,
                grid_dim = BLOCKS_PER_GRID, block_dim = THREADS_PER_BLOCK)
        ctx.synchronize()

        print("Result MatMul:")
        with output_devstor.map_to_host() as rs:
            for i in range(out_chans):
                print(rs[i], end = ", ")
            print()

