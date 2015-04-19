import matplotlib.pyplot as plt
import maxflow
import numpy as np
from os import listdir
import pdb
import random
from scipy.misc import imread, imsave
from scipy.spatial.distance import cdist
import time


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

def test_grabcut():
    box_list = sorted(['./bboxes/' + f for f in listdir('./bboxes')])
    data_list = sorted(['./data/' + f for f in listdir('./data')])
    truth_list = sorted(['./ground_truth/' + f for f in listdir('./ground_truth')])

    output = open('./results.txt','wb')
    mean_accuracy = 0.0

    for box_path, img_path, truth_path in zip(box_list, data_list, truth_list):
        with open(box_path) as box_file:
            bounds = box_file.readline().split()

            box = {
                'x_min': int(bounds[0]),
                'y_min': int(bounds[1]),
                'x_max': int(bounds[2]),
                'y_max': int(bounds[3])
            }

        print 'SEGMENTING IMAGE {}'.format(img_path)
        img = imread(img_path)

        ground_truth = imread(truth_path)
        ground_truth[ground_truth > 0] = 1

        start = time.time()
        seg = grabcut(img, box)
        end = time.time()
        print 'SEGMENTATION TOOK {} SECONDS FOR {} x {} IMAGE'.format(end - start, img.shape[1], img.shape[0])

        accuracy = get_accuracy(seg, ground_truth)
        mean_accuracy += accuracy
        
        similarity = 0.0
        #similarity = jaccard_similarity(seg, ground_truth)

        print 'OBTAINED FINAL ACCURACY: {}, JACCARD SIMILARITY: {}'.format(accuracy, similarity)
        imsave('./output/' + img_path.split('/')[2], np.where(seg == 0, 0, 255))
        output.write('{}\t{}\n'.format(box_path, similarity))

    mean_accuracy /= len(data_list)
    print 'MEAN ACCURACY: {}'.format(mean_accuracy)

    output.close()

def get_accuracy(seg_map, ground_truth):
    correct = (seg_map + ground_truth - 1) ** 2
    return np.mean(correct)

def grabcut(img, box=None):
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
        seg_map, fg_ass, bg_ass = estimate_segmentation(img, fg_gmm, bg_gmm, seg_map, box)
        fg = img[seg_map == 1]
        bg = img[seg_map == 0]

        print 'AT ITERATION {}, {} FOREGROUND / {} BACKGROUND'.format(i + 1, fg.shape[0], bg.shape[0])

    return seg_map

def preprocess(img):
    return img

def estimate_segmentation(img, fg_gmm, bg_gmm, seg_map, box):
    # Calculate unary values and component assignments
    fg_unary, fg_ass = get_unary(img, fg_gmm)
    bg_unary, bg_ass = get_unary(img, bg_gmm)

    # Calculate pairwise values
    pair_pot = get_pairwise(img)

    # Remove portion outside bounding box
    fg_unary = fg_unary[box['y_min']:box['y_max'], box['x_min']:box['x_max']]
    bg_unary = bg_unary[box['y_min']:box['y_max'], box['x_min']:box['x_max']]
    pair_pot = pair_pot[:, box['y_min']:box['y_max'], box['x_min']:box['x_max']]

    # Construct graph and run mincut on it
    pot_graph, nodes = create_graph(fg_unary, bg_unary, pair_pot)
    pot_graph.maxflow()

    # Create bit map of result and return it
    box_seg = segment(pot_graph, nodes)
    
    seg_map = np.zeros((img.shape[0], img.shape[1]), dtype='int32')
    seg_map[box['y_min']:box['y_max'], box['x_min']:box['x_max']] = box_seg

    return seg_map, fg_ass[seg_map == 1], bg_ass[seg_map == 0]

def adjust_outside_box(fg_unary, bg_unary, box):
    fg_unary[:box['y_min'], :] = -100000
    bg_unary[:box['y_min'], :] = 100000

    fg_unary[:, :box['x_min']] = -100000
    bg_unary[:, :box['x_min']] = 100000

    fg_unary[box['y_max']:, :] = -100000
    bg_unary[box['y_max']:, :] = 100000

    fg_unary[:, box['x_max']:] = -100000
    bg_unary[:, box['x_max']:] = 100000

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
    return (np.int_(np.logical_not(segments)) - 1) * -1

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
    log_pdfs = np.zeros((K, H, W), dtype='float64')

    for k, gaussian in enumerate(gmm):
        cov = gaussian['cov']
        mu_img = img - np.reshape(gaussian['mean'], (1, 1, 3))

        log_pdfs[k] += np.log(1.0 / np.sqrt(2 * (np.pi ** 3) * np.linalg.det(cov)))

        piece1 = -np.log(gaussian['size']) + 0.5 * np.log(np.linalg.det(cov))
        temp = np.einsum('ijk,il', np.transpose(mu_img), np.linalg.inv(cov))
        
        piece2 = np.zeros((H, W))
        
        for i in xrange(H):
            for j in xrange(W):
                piece2[i, j] = np.dot(temp[j, i], mu_img[i, j])
        
        potentials[k] = piece1 + piece2
        log_pdfs[k] += -1.0 * piece2
    
    try:
        unary = np.max(np.array(potentials), axis=0)
        assignments = np.argmax(np.array(log_pdfs), axis=0)
    except:
        pdb.set_trace()

    print 'CALCULATING UNARY POTENTIALS...'

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

        # print "AVERAGE CENTROID CHANGE: {}".format(avg_centroid_change)

        fg_mu = new_fg_mu
        bg_mu = new_bg_mu

    return fg_ass, bg_ass

def fit_gmm(fg, bg, fg_ass, bg_ass, k=5):
    fg_gmms, bg_gmms = [], []

    for i in xrange(k):
        fg_cluster = fg[fg_ass == i]
        bg_cluster = bg[bg_ass == i]

        # print 'FG CLUSTER SIZE: {}'.format(fg_cluster.shape[0])
        # print 'BG CLUSTER SIZE: {}'.format(bg_cluster.shape[0])

        fg_gmm, bg_gmm = {}, {}

        fg_gmm['mean'] = np.mean(fg_cluster, axis=0)
        bg_gmm['mean'] = np.mean(bg_cluster, axis=0)

        fg_gmm['cov'] = np.cov(fg_cluster, rowvar=0) + np.identity(3)*1e-8
        bg_gmm['cov'] = np.cov(bg_cluster, rowvar=0) + np.identity(3)*1e-8

        fg_gmm['size'] = fg_cluster.shape[0] * 1.0 / fg.shape[0]
        bg_gmm['size'] = bg_cluster.shape[0] * 1.0 / bg.shape[0]

        if fg_gmm['size'] > 0.001:
            fg_gmms.append(fg_gmm)
        if bg_gmm['size'] > 0.001:
            bg_gmms.append(bg_gmm)

    print 'FITTING GMM...'

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
        test_grabcut()
