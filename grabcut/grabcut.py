import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy.misc import imread
from scipy.spatial.distance import cdist
import random
import scipy


KMEANS_CONVERGENCE = 1.0

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
    
    # Initialize GMM components with k-means
    initial = quick_k_means(img, seg_map)
    scipy.misc.imsave('/Users/Jasper/Desktop/test.png',initial)
    gmm = fit_gmm(img, seg_map)

    return seg_map

def preprocess(img):
    return img

def quick_k_means(img, seg_map, k=5):
    foreground = img[seg_map == 1]
    background = img[seg_map == 0]

    fg_mu = foreground[np.random.choice(foreground.shape[0], k), :]
    bg_mu = background[np.random.choice(background.shape[0], k), :]
    avg_centroid_change = float('Inf')

    while avg_centroid_change > KMEANS_CONVERGENCE:
        fg_dist = cdist(foreground, fg_mu, metric='euclidean')
        bg_dist = cdist(background, bg_mu, metric='euclidean')

        fg_ass = np.argmin(fg_dist, axis=1)
        bg_ass = np.argmin(bg_dist, axis=1)

        new_fg_mu = np.zeros_like(fg_mu)
        new_bg_mu = np.zeros_like(bg_mu)

        for i in xrange(k):
            new_fg_mu[i] = np.mean(foreground[fg_ass == i], axis=0)
            new_bg_mu[i] = np.mean(background[bg_ass == i], axis=0)

        avg_centroid_change = np.sqrt(np.sum(np.square(fg_mu - new_fg_mu), axis=1))
        avg_centroid_change += np.sqrt(np.sum(np.square(fg_mu - new_fg_mu), axis=1))
        avg_centroid_change = np.mean(avg_centroid_change) / 2

        print "AVERAGE CENTROID CHANGE: {}".format(avg_centroid_change)

        fg_mu = new_fg_mu
        bg_mu = new_bg_mu

    return fg_ass, bg_ass


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
    points = len(X)
    color = [ ]
    for x in X:
        color.append(np.array(x[0]))
    colorList = np.array(color)
    colors = np.zeros((colorList.shape[0],3,len(mu)))
    mus = np.zeros_like(colors)
    for i in range(len(mu)):
        colors[:,:,i] = colorList
        mus[:,:,i] = np.tile(mu[i],(points,1))
    difference = np.sum(np.square(colors - mus),axis = 1)
    index = np.argmin(difference,axis=1)
    clusters1 = { }
    clusters2 = { }
    for j,k,z in zip(index,range(points),color):
        try:
            clusters1[tuple(mu[j])].append(k)
            clusters2[tuple(mu[j])].append(z)
        except KeyError:
            clusters1[tuple(mu[j])] = [k]
            clusters2[tuple(mu[j])] = [z]
    return clusters1, clusters2

def reevaluate_centers(clusters):
    newmu = [ ]
    for k in clusters:
        total = np.zeros((1,3))
        for a in clusters[k]:
            total += a
        total /= len(clusters[k])
        newmu.append(total)
    return [a[0] for a in newmu]

def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])

def k_means(img, box, k=3):
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
        clusters1, clusters2 = cluster_points(colorList, mu)
        #reevaluate centers
        mu = reevaluate_centers(clusters2)
    initial = np.zeros(img.shape)
    width = x_max - x_min
    for i,j in enumerate(clusters1.keys()):
        for k in clusters1[j]:
            initial[int(k/width)+y_min,k%width+x_min] = (i+1)*30
    return initial

def fit_gmm(img, seg_map, k=5):
    pass

def jaccardSimiliarity(segmentation,truth):
    truth = np.where(truth == 255, 1, 0)
    total = segmentation + truth
    intersection = np.sum(np.sum(np.where(total == 2, 1, 0)))
    union = np.sum(np.sum(np.where(total == 1 or total == 2, 1, 0)))
    return float(intersection)/float(union)*100

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
