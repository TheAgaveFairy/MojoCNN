from lenet import ftype


fn showProgress(progress: Int, total: Int) -> None:
    alias bar_width = 50
    var ratio = progress / total
    var filled = Int(bar_width * ratio)
    # print(chr(27) + "[2J",end="")
    print("\r[", end="")
    for _ in range(filled):
        print("=", end="")
    for _ in range(filled, bar_width):
        print(" ", end="")
    print("]", round(ratio * 100, 3), "%", end="")


@always_inline
fn reLu(x: Scalar[ftype]) -> Scalar[ftype]:
    # TODO: pass around as parameter for CPU
    return x if x > 0 else 0


@always_inline
fn reLuGrad(y: Scalar[ftype]) -> Scalar[ftype]:
    # TODO: Make this a function that we pass around for CPU
    return 1 if y > 0 else 0
