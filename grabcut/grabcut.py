import matplotlib.pyplot as plt
import numpy as np
import pdb
import random
import scipy
import sys
from scipy.misc import imread
from scipy.spatial.distance import cdist


KMEANS_CONVERGENCE = 1.0
MAX_NUM_ITERATIONS = 5

GAMMA = 50

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
    
    fg = img[seg_map == 1]
    bg = img[seg_map == 0]

    # Initialize GMM components with k-means
    fg_ass, bg_ass = quick_k_means(fg, bg)
    
    # Iteratively refine segmentation from initialization
    for i in xrange(MAX_NUM_ITERATIONS):
        fg_gmm, bg_gmm = fit_gmm(fg, bg, fg_ass, bg_ass)
        seg_map, fg_ass, bg_ass = estimate_segmentation(img, fg_gmm, bg_gmm, seg_map)

        fg = img[seg_map == 1]
        bg = img[seg_map == 0]

    return seg_map

def preprocess(img):
    return img

def estimate_segmentation(img, fg_gmm, bg_gmm, seg_map):
    # Calculate unary values and component assignments
    fg_unary, fg_ass = get_unary(img, fg_gmm)
    bg_unary, bg_ass = get_unary(img, bg_gmm)

    # Calculate pairwise values
        # Use subimages shifted up, down, left, right
    pair_pot = get_pairwise(img)
    # Construct graph and run mincut on it
    # Create bit map of result and return it

    return seg_map, fg_ass, bg_ass

def get_pairwise(img):
    H, W, C = img.shape

    shifted_imgs = shift(img)
    pairwise_dist = np.zeros((4, H, W, C))

    for i in xrange(4):
        pairwise_dist[i] = np.sum((img - shifted_imgs[i]) ** 2, axis=2)

    beta = 1.0 / (2 * np.mean(pairwise_dist))

    pairwise_dist = np.exp(-1 * beta * pairwise_dist)
    pairwise_dist *= GAMMA

    return pairwise_dist

def shift(img):
    H, W, C = img.shape

    up = np.array(img)
    up[:H-1, : ,:] = img[1:, :, :]

    down = np.array(img)
    down[1:, :, :] = img[:H-1, :, :]

    left = np.array(img)
    left[:, :W-1, :] = img[:, 1:, :]

    right = np.array(img)
    left[:, 1:, :] = img[:, :W-1, :]

    shifted_imgs = np.zeros((4, H, W, C))
    shifted_imgs[0] = up
    shifted_imgs[1] = down
    shifted_imgs[2] = left
    shifted_imgs[3] = right

    return shifted_imgs

def get_unary(img, gmms):
    # Find closest component in gmm set
    # Calculate log PDF
    #determine most likely component
    x = tuple(map(tuple,img))
    imgData = [ ] 
    for a in x:
        imgData.extend(a)
    maximum = [ ]
    for components in gmms:
        cov = components['cov']
        mu = components['mean']
        weight = components['size']
        piece1 = -np.log(weight)+0.5*np.log(np.linalg.det(cov))
        piece2 = np.dot(imgData - mu,np.linalg.inv(cov))
        temp = [ ]
        for i, j in zip(piece2, imgData - mu):
            temp.append(np.dot(i,np.transpose(j)))
        maximum.append(piece1+temp)
    unary = np.max(np.array(maximum),axis = 0)
    assign = [ ]
    assignments = np.argmax(np.array(maximum),axis = 0)
    for elements in assignments:
        assign.append(gmms[elements]['mean'])
    return unary, assign

def quick_k_means(foreground, background, k=5):
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

def fit_gmm(fg, bg, fg_ass, bg_ass, k=5):
    fg_gmms, bg_gmms = [], []

    for i in xrange(k):
        fg_cluster = fg[fg_ass == i]
        bg_cluster = bg[bg_ass == i]

        fg_gmm, bg_gmm = {}, {}

        fg_gmm['mean'] = np.mean(fg_cluster, axis=0)
        bg_gmm['mean'] = np.mean(bg_cluster, axis=0)

        fg_gmm['cov'] = np.cov(fg_cluster, rowvar=0) + np.identity(3)*1e-8
        bg_gmm['cov'] = np.cov(bg_cluster, rowvar=0) + np.identity(3)*1e-8

        fg_gmm['size'] = fg_cluster.shape[0] * 1.0 / fg.shape[0]
        bg_gmm['size'] = bg_cluster.shape[0] * 1.0 / bg.shape[0]

        fg_gmms.append(fg_gmm)
        bg_gmms.append(bg_gmm)

    return fg_gmms, bg_gmms

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

def cluster_points(color, mu):
    points = len(color)
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

def k_means(img, box, k=5):
    y_min = box['y_min']
    y_max = box['y_max']
    x_min = box['x_min']+1
    x_max = box['x_max']
    colorList = [ ]
    for y in range(int(y_min),int(y_max)):
        for x in range(int(x_min),int(x_max)):
            colorList.append(img[y,x,:])
    oldmu = [a for a in random.sample(colorList,k)]
    mu = [a for a in random.sample(colorList,k)]
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

def jaccard_similarity(segmentation,truth):
    truth = np.where(truth == 255, 1, 0)
    total = segmentation + truth
    intersection = np.sum(np.sum(np.where(total == 2, 1, 0)))
    union = np.sum(np.sum(np.where(total == 1 or total == 2, 1, 0)))
    return float(intersection)/float(union)*100

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        grabcut(sys.argv[1])
    else:
        grabcut("./data/banana1.bmp")
