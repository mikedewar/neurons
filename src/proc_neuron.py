import cv
import numpy as np
from tiffstack import TiffStack

neuron = TiffStack("../img/neurons1.tif", scale=0.5)
seq = [im for im in neuron]
seq.pop(0) # the first image starts at 1...
a = np.array(seq)
b = a.sum(0)
v = a.reshape(a.shape[1]*a.shape[2],a.shape[0])
