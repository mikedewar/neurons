import cv
import numpy as np

# this finds the blobs in ../cache/neurons_mask/tif
# here a blob is defined as a contiguous region of non-zero pixels

def find_neighbours(point, max_i, max_j):
    """
    this finds all the neighbours of the point that are 
    within the image
    """
    i, j = point
    n = []
    if j < max_j:
        n.append((i, j + 1))
    if j > 0:
        n.append((i, j - 1))
    if i < max_i:
        n.append((i + 1, j))
    if i > 0:
        n.append((i - 1, j))
    return n

def grow(point, i, ID, mask):
    """
    this is a recursive algorithm that builds up the ID array
    while destroying the mask array.
    """
    ID[point] = i
    mask[point] = 0
    neighbours = find_neighbours(point, mask.shape[0] - 1, mask.shape[1] - 1)
    for p in neighbours:
        if mask[p]:
            grow(p, i, ID, mask)

def find_blobs(mask_filename='../cache/neurons_mask.tif'):
    """
    This takes the mask image and returns an array 
    containing the ID of each pixel
    
    the algorithm expects the blobs to be black (colour=0) 
    and the rest to be white (color = 1)
    """
    mask_im = cv.LoadImageM(
        mask_filename,
        iscolor=cv.CV_LOAD_IMAGE_UNCHANGED
    )
    mask = np.asarray(mask_im, dtype=int)

    mask[np.where(mask == 0)] = 1
    mask[np.where(mask == 255)] = 0
    assert len(np.unique(mask)) == 2

    ID = np.zeros(mask.shape, dtype=int)
    i = 0
    while mask.sum():
        a, b = np.where(mask)
        point = (a[0], b[0])
        i += 1
        grow(point, i, ID, mask)

    return ID


if __name__ == "__main__":
    ID = find_blobs()
    np.save('../cache/neuron_identities.npy', ID)

