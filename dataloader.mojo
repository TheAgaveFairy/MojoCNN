from image import Image
import os

struct MNISTDataRepository:
    alias COUNT_TRAIN =     60000
    alias COUNT_TEST =      10000

    var data_dir: String
    var train_image_file: String
    var train_label_file: String
    var test_image_file: String
    var test_label_file: String

    fn __init__(out self, data_dir: String = "data"):
        self.data_dir = data_dir
        self.train_image_file = data_dir + "/train-images-idx3-ubyte"
        self.train_label_file = data_dir + "/train-labels-idx1-ubyte"
        self.test_image_file = data_dir + "/t10k-images-idx3-ubyte"
        self.test_label_file = data_dir + "/t10k-labels-idx1-ubyte"

    fn loadTrainingData(self, count: UInt, ptr: UnsafePointer[Image]) raises -> None:
        self._readData(count, "train", ptr)

    fn loadTestingData(self, count: UInt, ptr: UnsafePointer[Image]) raises -> None:
        self._readData(count, "test", ptr)

    fn _readData(self, count: UInt, test_or_train: String, ptr: UnsafePointer[Image]) raises -> None:
        var data_filename = self.test_image_file if test_or_train == "test" else self.train_image_file
        var label_filename = self.test_label_file if test_or_train == "test" else self.train_label_file
        try:
            var data_file = open(data_filename, "r")
            var label_file = open(label_filename, "r")
        
            _ = data_file.seek(16, os.SEEK_SET)    # data has an unknown header
            _ = label_file.seek(8, os.SEEK_SET)  # labels too

            alias buffer_size = Image.PixelLayout.size() #IMAGE_SIZE * IMAGE_SIZE
            var image_buffer = UnsafePointer[UInt8].alloc(buffer_size)
            
            for c in range(count):
                var data_list = data_file.read_bytes(buffer_size)
                
                var temp = label_file.read_bytes(1)
                var data_label: UInt8 = temp[0]

                @parameter
                for i in range(Image.PixelTensor.shape[0]()):
                    @parameter
                    for j in range(Image.PixelTensor.shape[1]()):
                        idx = i * Image.PixelTensor.shape[0]() + j
                        image_buffer[idx] = data_list[idx]

                var test_image = Image(image_buffer, data_label)
                ptr[c] = test_image

            image_buffer.free()
            data_file.close()
            label_file.close()
        except e:
            print("Error with input binary files:", e)

    @staticmethod
    fn shuffleData(data: UnsafePointer[Image], count: Int, seed: Int = 69):
        """
        Not needed, but I / Claude wrote it just to play around and learn.
        """
        if count < 1:
            return
        var rng_state = seed
        #some Claude 4 work
        for i in range(count - 1, 0, -1):
            rng_state = (rng_state * 1664525 + 1013904223) % 2147483647
            var j = Int(rng_state) % (i + 1)

            var temp = data[i]
            data[i] = data[j]
            data[j] = temp
