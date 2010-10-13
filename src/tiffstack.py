import Image
import numpy as np

class TiffStack:
    """
    this provides a class that makes splitting up tiffs nice'n'easy
    """
    def __init__(self, filename):
        self.im = Image.open(filename)

    def __getitem__(self, ix):
        try:
            if ix:
                self.im.seek(ix)
                img_size = (self.im.size[1], self.im.size[0])
                return np.array(self.im.getdata(),
                                dtype=np.uint16).reshape(img_size)
        except EOFError:
            raise IndexError
