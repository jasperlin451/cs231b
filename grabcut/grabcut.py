import matplotlib.pyplot as plt
import maxflow
import numpy as np
import pdb
import random
import scipy
from os import listdir
from scipy.misc import imread
from scipy.misc import imsave
from scipy.spatial.distance import cdist


KMEANS_CONVERGENCE = 1.0
MAX_NUM_ITERATIONS = 5

GAMMA = 50

SHIFT_DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

STRUCTURES = {
    'UP': np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    'DOWN': np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    'LEFT': np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    'RIGHT': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
}
def runAlgorithm( ):
    bboxes = [f for f in listdir('./bboxes')]
    data = [f for f in listdir('./data')]
    ground_truth = [f for f in listdir('./ground_truth')]
    output = open('./results.txt','wb')
    for box, img, truth in zip(bboxes,data,ground_truth):
        f = open('./bboxes/'+box) 
        bounds = f.readline().split()
        bounding = { }
        bounding['xmin'] = int(bounds[0])
        bounding['ymin'] = int(bounds[1])
        bounding['xmax'] = int(bounds[2])
        bounding['ymax'] = int(bounds[3])
        image = imread('./data/'+img)
        gtruth = imread('./ground_truth/'+truth)
        seg = grabcut(image, bounding)
        similarity = jaccard_similarity(seg, gtruth)
        imsave('./output/'+img,np.where(seg==0,0,255))
        output.write(box+'\t'+similarity+'\n')
    output.close()
def grabcut(image_file, box=None):
    img = image_file
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
    pair_pot = get_pairwise(img)

    # Construct graph and run mincut on it
    pot_graph, nodes = create_graph(fg_unary, bg_unary, pair_pot)
    pot_graph.maxflow()

    # Create bit map of result and return it
    seg_map = segment(pot_graph, nodes)

    return seg_map, fg_ass, bg_ass

def create_graph(fg_unary, bg_unary, pair_pot):
    graph = maxflow.Graph[float]()
    nodes = graph.add_grid_nodes(fg_unary.shape)

    # Add unary potentials
    graph.add_grid_tedges(nodes, fg_unary, bg_unary)

    # Add pairwise potentials
    for i, direction in enumerate(SHIFT_DIRECTIONS):
        graph.add_grid_edges(nodes, weights=pair_pot[i], structure=STRUCTURES[direction], symmetric=False)

    return graph, nodes

def segment(graph, nodes):
    segments = graph.get_grid_segments(nodes)
    return np.int_(np.logical_not(segments))

def get_pairwise(img):
    H, W, C = img.shape

    shifted_imgs = shift(img)
    pairwise_dist = np.zeros((4, H, W))

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
    right[:, 1:, :] = img[:, :W-1, :]

    shifted_imgs = np.zeros((4, H, W, C), dtype='uint32')
    shifted_imgs[0] = up
    shifted_imgs[1] = down
    shifted_imgs[2] = left
    shifted_imgs[3] = right

    return shifted_imgs

def get_unary(img, gmm):
    K = len(gmm)
    H, W, C = img.shape

    potentials = np.zeros((K, H, W))

    for k, gaussian in enumerate(gmm):
        cov = gaussian['cov']
        mu_img = img - np.reshape(gaussian['mean'], (1, 1, 3))

        piece1 = -np.log(gaussian['size']) + 0.5 * np.log(np.linalg.det(cov))
        temp = np.einsum('ijk,il', np.transpose(mu_img), np.linalg.inv(cov))
        
        piece2 = np.zeros((H, W))
        
        for i in xrange(H):
            for j in xrange(W):
                piece2[i, j] = np.dot(temp[j, i], mu_img[i, j])
        
        potentials[k] = piece1 + piece2
    
    unary = np.max(np.array(potentials), axis=0)
    assignments = np.argmax(np.array(potentials), axis=0)

    return unary, assignments

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

#    pdb.set_trace()

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
        runAlgorithm()
