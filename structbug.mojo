from layout import Layout, LayoutTensor, print_layout

alias dtype = DType.float32
struct Test(Copyable, Movable):
    alias layout = Layout.row_major(4,5,6)
    var storage: UnsafePointer[Scalar[dtype]]
    var tensor: LayoutTensor[mut = True, DType.float32, Self.layout, MutableAnyOrigin]

    fn __init__(out self):
        self.storage = __type_of(self.storage).alloc(Self.layout.size())
        self.tensor = __type_of(self.tensor)(self.storage).fill(0)

    fn __copyinit__(out self, other: Self):
        self.storage = other.storage
        self.tensor = other.tensor

    #fn __moveinit__(out 

    @staticmethod
    fn tester() -> Self:
        var model = Self()
        print(model.tensor)
        for i in range(model.tensor.shape[0]()):
            for j in range(model.tensor.shape[1]()):
                for k in range(model.tensor.shape[2]()):
                    model.tensor[i, j, k] = i + j + k

        print(model.tensor)
        return model^

def main():
    var model = Test.tester()
    print(model.tensor)
