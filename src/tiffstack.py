import Image
import numpy as np

class TiffStack:
    def __init__(self,filename,scale=1):
        self.im = Image.open(filename)
        self.scale = scale
    
    def __getitem__(self, ix):
        try:
            if ix:
                self.im.seek(ix)
                # scale the image
                #scale_size = tuple([int(s*self.scale) for s in self.im.size])
                #im_scaled = self.im.resize(scale_size)
                # the size of the (scaled) image is width, height. 
                # We need num rows, num cols for when we convert to an array
                #img_size = (im_scaled.size[1],im_scaled.size[0])
                #return np.array(im_scaled.getdata(),dtype=np.uint16).reshape(img_size)
                img_size = (self.im.size[1],self.im.size[0])
                return np.array(self.im.getdata(),
                                dtype=np.uint16).reshape(img_size)

        except EOFError:
            raise IndexError
