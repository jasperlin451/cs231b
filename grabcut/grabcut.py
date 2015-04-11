import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.misc import imread

def grabcut(image_file, box=None):
    print 'LOADING IMAGE FROM FILE: {}'.format(image_file)
    img = imread(image_file)
    print 'IMAGE SHAPE: {}'.format(img.shape)

    # Allow user to select bounding box if not provided
    if box is None:
        box = select_bounding_box(img)
    print 'BOUNDING BOX COORDS: {}'.format(box)

    img = preprocess(img)

    # Initialize segmentation map to bounding box
    seg_map = np.zeros((img.shape[0], img.shape[1]))
    seg_map[(box['y_min'] + 1):box['y_max'], (box['x_min'] + 1):box['x_max']] = 1

    gmm = fit_gmm(img, seg_map)

    return seg_map

def preprocess(img):
    return img

def select_bounding_box(img):
    plt.imshow(img)

    click = plt.ginput(2)
    plt.close()
    
    box = {
        'x_min': round(min(click[0][0], click[1][0])),
        'y_min': round(min(click[0][1], click[1][1])),
        'x_max': round(max(click[0][0], click[1][0])),
        'y_max': round(max(click[0][1], click[1][1])),
    }
    
    width = float(img.shape[1])
    x_range_min = box['x_min'] / width
    x_range_max = box['x_max'] / width

    plt.axhspan(box['y_min'], box['y_max'], x_range_min, x_range_max, color='r', fill=False)
    plt.imshow(img)
    plt.show()

    return box

def fit_gmm(img, seg_map, k=5):
    pass

# INITIALIZE THE FOREGROUND & BACKGROUND GAUSSIAN MIXTURE MODEL (GMM)
# 
# while CONVERGENCE
#     
#     UPDATE THE GAUSSIAN MIXTURE MODELS
#     
#     MAX-FLOW/MIN-CUT ENERGY MINIMIZATION
#     
#     IF THE ENERGY DOES NOT CONVERGE
#         
#         break;

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        grabcut(sys.argv[1])
    else:
        grabcut("./data/banana1.bmp")