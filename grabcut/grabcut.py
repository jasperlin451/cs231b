import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.misc import imread
import random
import scipy
def grabcut(image_file, box=None):
    print 'LOADING IMAGE FROM FILE: {}'.format(image_file)
    img = imread(image_file)
    print 'IMAGE SHAPE: {}'.format(img.shape)

    # Allow user to select bounding box if not provided
    if box is None:
        box = select_bounding_box(img)
    print 'BOUNDING BOX COORDS: {}'.format(box)

    img = preprocess(img)

    # Initialize segmentation map to bounding box, foreground = 1, background = 0
    seg_map = np.zeros((img.shape[0], img.shape[1]))
    seg_map[(box['y_min'] + 1):box['y_max'], (box['x_min'] + 1):box['x_max']] = 1
    # initialize components with k-means, from lecture notes
    initial = k_means(img,box)
    scipy.misc.imsave('/Users/Jasper/Desktop/test.png',initial)
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

def cluster_points(X, mu):
    clusters = {}
    for x in X:
        color = x[0]
        dist = [ ]
        for i in mu:
            dist.append((np.linalg.norm(color-i),i))
        bestmukey = min(dist, key=lambda t:t[0])[1]
        try:
            clusters[tuple(bestmukey)].append(x)
        except KeyError:
            clusters[tuple(bestmukey)] = [x]
    return clusters

def reevaluate_centers(clusters):
    newmu = [ ]
    for k in clusters:
        total = np.zeros((1,3))
        for a in clusters[k]:
            total += a[0]
        total /= len(clusters[k])
        newmu.append(total[0])
    return newmu

def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])

def k_means(img, box, k=5):
    y_min = box['y_min']
    y_max = box['y_max']
    x_min = box['x_min']+1
    x_max = box['x_max']
    colorList = [ ]
    for y in range(int(y_min),int(y_max)):
        for x in range(int(x_min),int(x_max)):
            colorList.append((img[y,x,:],(y,x)))
    oldmu = [a[0] for a in random.sample(colorList,k)]
    mu = [a[0] for a in random.sample(colorList,k)]
    counter = 1
    while not has_converged(mu, oldmu):
        print counter
        counter += 1
        oldmu = mu
        #assign points to clusters
        clusters = cluster_points(colorList, mu)
        #reevaluate centers
        mu = reevaluate_centers(clusters)
    initial = np.zeros(img.shape)
    for i,j in enumerate(clusters.keys()):
        for k in clusters[j]:
            initial[k[1]] = (i+1)*30
    return initial

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
