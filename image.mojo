from layout import Layout, LayoutTensor, print_layout
from math import sqrt
from memory import memcpy

from lenet import IMAGE_SIZE, PADDED_SIZE, PADDING, ftype

struct Image(Stringable, Copyable):
    """
    I made the decision to store the raw pixels as they come from a file. Before
    loading into the features to start a pass, it needs to be in a normalized
    format. I didn't know which to store but didn't need both.
    """
    alias PixelLayout = Layout.row_major(IMAGE_SIZE, IMAGE_SIZE)
    alias PixelStorage = InlineArray[UInt8, Self.PixelLayout.size()]
    alias PixelTensor = LayoutTensor[mut = True, DType.uint8, Self.PixelLayout, MutableAnyOrigin]

    alias DataLayout = Layout.row_major(PADDED_SIZE, PADDED_SIZE)
    alias DataStorage = InlineArray[Scalar[ftype], Self.DataLayout.size()] # stack won't work
    alias DataTensor = LayoutTensor[mut = True, ftype, Self.DataLayout, MutableAnyOrigin]
    
    var pixel_storage: UnsafePointer[UInt8]
    var pixels: Self.PixelTensor
    var label: UInt8 # digits [0, 9] MNIST

    fn __init__(out self, ptr: UnsafePointer[UInt8], label: UInt8):
        if label > 9:
            print("Error with incoming label for image:", label)
        self.pixel_storage = UnsafePointer[UInt8].alloc(Self.PixelLayout.size())
        self.pixels = Self.PixelTensor(self.pixel_storage)
        # memcpy possible
        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                var idx = r * IMAGE_SIZE + c
                self.pixels[r, c] = ptr[idx]

        self.label = label

    # i decided to free things at load (see toNormalized())
    #fn __del__(owned self):
        #self.pixels.ptr.free()

    fn toNormalized(self) -> Self.DataTensor:
        # TODO: check where / if storage gets freed
        """
        Normalizes from 28x28 uint8 to zero-padded 32x32 float32 (or whatever ftype is for the model).
        
        Memory not freed! Might want to clarify that.
        """
        #mut = False gives a terrible terrible compiler warning as an aside for making LayoutTensors
        var storage = UnsafePointer[Scalar[ftype]].alloc(Self.DataLayout.size())
        #var storage = DataStorage() # needs to be on the heap
        var tensor = Self.DataTensor(storage).fill(0.0)

        var mean: Float64
        var std: Float64

        var sum: UInt64 = 0
        var std_sum: UInt64 = 0
        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                sum += UInt(self.pixels[r, c]) # SIMD possible?
                std_sum += Int(self.pixels[r, c].cast[DType.uint64]() * self.pixels[r, c].cast[DType.uint64]()) # rebind[UInt64]
        
        alias num_elems = IMAGE_SIZE * IMAGE_SIZE
        mean = Float64(sum) / num_elems
        var temp = Float64(std_sum) / num_elems - mean * mean
        std = sqrt(temp)

        for r in range(IMAGE_SIZE):
            for c in range(IMAGE_SIZE):
                var curr = Float64(Int(self.pixels[r, c]))
                tensor[r + PADDING, c + PADDING] = ((curr - mean) / std).cast[ftype]()
         
        return tensor

    fn __str__(self) -> String:
        """
        Pretty printing for debugging and fun.
        """
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
        self.pixel_storage = UnsafePointer[UInt8].alloc(Self.PixelLayout.size())
        memcpy(self.pixel_storage, other.pixel_storage, Self.PixelLayout.size())
        self.pixels = Self.PixelTensor(self.pixel_storage)
        self.label = other.label

    fn __moveinit__(out self, owned existing: Self):
        self.pixel_storage = existing.pixel_storage
        self.pixels = __type_of(self.pixels)(self.pixel_storage)
        existing.pixel_storage = __type_of(existing.pixel_storage)()
        self.label = existing.label


