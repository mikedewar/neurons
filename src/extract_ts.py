"""
This script extracts the mean intensity of every blob
stored in the id array and writes it to disk
"""

import numpy as np
from tiffstack import TiffStack

print "loading stack"
seq = [im for im in TiffStack("../data/neurons.tif")]
seq.pop(0) # the first image starts at 1...
id = np.load('../cache/neuron_identities.npy')

print "extracting time series"
num_blobs = len(np.unique(id)) - 1
ts = np.empty((num_blobs, len(seq)))
for i, A in enumerate(seq):
    for j in range(num_blobs):
        ts[j, i] = np.sum(A[id == (j + 1)]) / float(np.sum(id == (j + 1)))

print "saving time series"
np.save('../cache/neuron_ts', ts)
np.savetxt('../cache/neuron_ts.csv.gz', ts, delimiter=',', fmt='%-10.5f')
