from layout import Layout, LayoutTensor, print_layout
import os
from sys.info import num_logical_cores
from gpu.host import DeviceContext, DeviceFunction, DeviceBuffer, info, DeviceAttribute

alias dtype = DType.float32

def main():

    # Check if any CUDA devices are available
    var num_cuda_devices = DeviceContext.number_of_devices(api="cuda")
    print("Number of CUDA devices:", num_cuda_devices)
    
    #if num_cuda_devices > 0:
    for dev in range(num_cuda_devices):
        var ctx = DeviceContext(dev)

        var device_name = ctx.name()
        print("Device name:", device_name)
        
        # Get the maximum number of threads per block
        var max_threads = ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK)
        print("Maximum threads per block:", max_threads)

        var max_block_dims = (ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_X), ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_Y), ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_Z))
        print("Maximum block DIM for x, y, z:", max_block_dims[0], max_block_dims[1], max_block_dims[2])

        var max_grid_dims = (ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X), ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_Y), ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_Z))
        print("Maximum grid DIM for x, y, z:", max_grid_dims[0], max_grid_dims[1], max_grid_dims[2])

        # Get the warp size
        var warp_size = ctx.get_attribute(DeviceAttribute.WARP_SIZE)
        print("Warp size:", warp_size)
        
        var max_blocks_per_multiprocessor = ctx.get_attribute(DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR)

        # Get number of multiprocessors
        var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        print("Number of multiprocessors:", sm_count)
        
        # Get maximum threads per multiprocessor
        var threads_per_sm = max_threads * max_blocks_per_multiprocessor
        print("Maximum threads per multiprocessor:", threads_per_sm)
        
        # Calculate total threads
        print("Total potential threads:", sm_count * threads_per_sm)

        var max_sm_pb = ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK)
        print("Max shared mem per block:", max_sm_pb)

        var max_sm_pmp = ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
        print("Max shared mem per multiproc:", max_sm_pmp)

        print()
    #else: # that shouldnt be allowed
        #print("No CUDA devices found")
    
    # For comparison, show CPU threads
    print("CPU logical cores:", num_logical_cores())
