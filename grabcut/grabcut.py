import matplotlib.pyplot as plt
import maxflow
import numpy as np
from os import listdir
import pdb
import random
from scipy.misc import imread, imsave
from scipy.spatial.distance import cdist
import time

# TODO:
#  -look at decreasing number of components for background

KMEANS_CONVERGENCE = 1.0
MAX_NUM_ITERATIONS = 7
THRESHOLD = 0.01
GAMMA = 50

INITIAL_NUMBER_COMPONENTS = 4
NUMBER_FG_COMPONENTS = 4
NUMBER_BG_COMPONENTS = 4

SHIFT_DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UP_LEFT', 'UP_RIGHT']

STRUCTURES = {
    'UP': np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    'DOWN': np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    'LEFT': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    'RIGHT': np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    'UP_LEFT': np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
    'UP_RIGHT': np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
}

def test_grabcut():
    box_list = sorted(['./bboxes/' + f for f in listdir('./bboxes')])
    data_list = sorted(['./data/' + f for f in listdir('./data')])
    truth_list = sorted(['./ground_truth/' + f for f in listdir('./ground_truth')])
    output = open('./results.txt','wb')

    mean_accuracy = 0.0
    mean_similarity = 0.0

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

        #keep_going = True

        #while(keep_going):
        start = time.time()
        #try:
        seg = grabcut(img, box)
        #    keep_going = False
        #except:
        #    pass
        end = time.time()
        print 'SEGMENTATION TOOK {} SECONDS FOR {} x {} IMAGE'.format(end - start, img.shape[1], img.shape[0])

        accuracy = get_accuracy(seg, ground_truth)
        mean_accuracy += accuracy
        
        similarity = jaccard_similarity(seg, ground_truth)
        mean_similarity += similarity
        print 'OBTAINED FINAL ACCURACY: {}, JACCARD SIMILARITY: {}'.format(accuracy, similarity)
        out = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if seg[i,j] == 1:
                    out[i,j,:] = img[i,j,:]

        imsave('./output/' + img_path.split('/')[2], out)
        output.write('{}\t{}\n'.format(box_path, similarity))

    mean_accuracy /= len(data_list)
    mean_similarity /= len(data_list)
    print 'MEAN ACCURACY: {}, MEAN SIMILARITY: {}'.format(mean_accuracy,mean_similarity)

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
    seg_map[box['y_min'] - 1:box['y_max'] + 1, box['x_min'] - 1:box['x_max'] + 1] = 1
    
    fg = img[seg_map == 1]
    bg = img[seg_map == 0]

    # Initialize GMM components with k-means
    fg_ass, bg_ass = quick_k_means(fg, bg)

    # Iteratively refine segmentation from initialization
    change = 1
    i = 0
    oldshape = fg.shape[0]
    while change > THRESHOLD:
        fg_gmm, bg_gmm = fit_gmm(fg, bg, fg_ass, bg_ass)
        seg_map, fg_ass, bg_ass = estimate_segmentation(img, fg_gmm, bg_gmm, seg_map, box)

        fg = img[seg_map == 1]
        bg = img[seg_map == 0]
        
        change = abs(oldshape-fg.shape[0])/float(oldshape)
        oldshape = fg.shape[0]
        print 'AT ITERATION {}, {} FOREGROUND / {} BACKGROUND'.format(i + 1, fg.shape[0], bg.shape[0])
        print 'FOREGROUND ASSIGNMENT CHANGED {}%'.format(str(100.0 * change)[:5])
        i += 1
    return seg_map


def preprocess(img):
    return img

def adjust_outside_box(fg_unary, bg_unary, box):
    fg_unary[:box['y_min'], :] = 1e15
    bg_unary[:box['y_min'], :] = -1e15

    fg_unary[:, :box['x_min']] = 1e15
    bg_unary[:, :box['x_min']] = -1e15

    fg_unary[box['y_max']:, :] = 1e15
    bg_unary[box['y_max']:, :] = -1e15

    fg_unary[:, box['x_max']:] = 1e15
    bg_unary[:, box['x_max']:] = -1e15
    
    return fg_unary, bg_unary

def estimate_segmentation(img, fg_gmm, bg_gmm, seg_map, box):
    # Calculate unary values and component assignments
    fg_unary, fg_ass = get_unary(img, fg_gmm)
    bg_unary, bg_ass = get_unary(img, bg_gmm)

    # Calculate pairwise values
    pair_pot = get_pairwise(img)

    #pdb.set_trace()
    fg_unary,bg_unary = adjust_outside_box(fg_unary,bg_unary,box)
    # Construct graph and run mincut on it
    pot_graph, nodes = create_graph(fg_unary, bg_unary, pair_pot)
    pot_graph.maxflow()

    # Create bit map of result and return it
    box_seg = segment(pot_graph, nodes)
    
    return box_seg, fg_ass[box_seg == 1], bg_ass[box_seg == 0]

def create_graph(fg_unary, bg_unary, pair_pot):
    graph = maxflow.Graph[float]()
    nodes = graph.add_grid_nodes(fg_unary.shape)

    # Add unary potentials
    graph.add_grid_tedges(nodes, fg_unary, bg_unary)

    # Add pairwise potentials
    for i, direction in enumerate(SHIFT_DIRECTIONS):
        if i in [0, 2, 4, 5]:
            graph.add_grid_edges(nodes, weights=pair_pot[i], structure=STRUCTURES[direction], symmetric=False)

    return graph, nodes

def segment(graph, nodes):
    segments = graph.get_grid_segments(nodes)
    return (np.int_(np.logical_not(segments)) - 1) * -1

def get_pairwise(img):
    H, W, C = img.shape

    shifted_imgs = shift(img)
    pairwise_dist = np.zeros((6, H, W))

    for i in xrange(6):
        pairwise_dist[i] = np.sqrt(np.sum((img - shifted_imgs[i]) ** 2, axis=2))

    beta = 1.0 / (2 * np.mean(pairwise_dist[[0,2,4,5],:,:]))

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

    up_left = np.array(img)
    up_left[:H-1, :W-1, :] = img[1:, 1:, :]

    up_right = np.array(img)
    up_right[:H-1, 1:, :] = img[1:, :W-1, :]

    shifted_imgs = np.zeros((6, H, W, C), dtype='uint32')
    shifted_imgs[0] = up
    shifted_imgs[1] = down
    shifted_imgs[2] = left
    shifted_imgs[3] = right
    shifted_imgs[4] = up_left
    shifted_imgs[5] = up_right

    return shifted_imgs

def get_unary(img, gmm):
    K = len(gmm)
    H, W, C = img.shape

    potentials = np.zeros((K, H, W))
    log_pdfs = np.zeros((K, H, W), dtype='float64')

    for k, gaussian in enumerate(gmm):
        cov = gaussian['cov']
        mu_img = img - np.reshape(gaussian['mean'], (1, 1, 3))

        log_pdfs[k] +=  -0.5*np.log(np.linalg.det(cov))

        piece1 = -np.log(gaussian['size']) + 0.5 * np.log(np.linalg.det(cov))
        temp = np.einsum('ijk,il', np.transpose(mu_img), np.linalg.inv(cov))
        
        piece2 = np.zeros((H, W))
        
        for i in xrange(H):
            for j in xrange(W):
                piece2[i, j] = np.dot(temp[j, i], mu_img[i, j])

        piece2 *= 0.5
        potentials[k] = piece1 + piece2
        log_pdfs[k] += -1.0 * piece2
    
    assignments = np.argmax(np.array(log_pdfs), axis=0)
    unary = np.zeros((H,W))
    for i in xrange(H):
        for j in xrange(W):
            unary[i,j] = potentials[assignments[i,j],i,j]

    return unary, assignments

def quick_k_means(foreground, background, k=INITIAL_NUMBER_COMPONENTS):
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

def fit_gmm(fg, bg, fg_ass, bg_ass, k=INITIAL_NUMBER_COMPONENTS):
    fg_gmms, bg_gmms = [], []

    for i in xrange(max(np.max(fg_ass), np.max(bg_ass)) + 1):
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

    if len(fg_gmms) < k:
        split(fg_gmms, fg, fg_ass, k)
    #if len(bg_gmms) < k:
    #    split(bg_gmms, bg, bg_ass, k)

    """
    if len(fg_gmms) < NUMBER_FG_COMPONENTS:
        print 'REINITIALIZING FOREGROUND COMPONENTS...'
        fg_ass, temp = quick_k_means(fg, fg, k=NUMBER_FG_COMPONENTS)
        return fit_gmm(fg, bg, fg_ass, bg_ass)
    """
    """
    if len(bg_gmms) < NUMBER_BG_COMPONENTS:
        print 'REINITIALIZING BACKGROUND COMPONENTS...'
        bg_ass, temp = quick_k_means(bg, bg, k=NUMBER_BG_COMPONENTS)
        return fit_gmm(fg, bg, fg_ass, bg_ass)
    """

    return fg_gmms, bg_gmms

def split(gmm_list, pixels, assignment, k):
    sizes = np.array([f['size'] for f in gmm_list])
    orig_size = np.max(sizes)
    gmm_list.pop(np.argmax(sizes))

    largest = np.argmax(np.bincount(assignment))
    members = pixels[assignment == largest]

    num_new_comps = k - len(gmm_list)
    ass1, ass2 = quick_k_means(members, members, k=num_new_comps)

    for i in xrange(num_new_comps):
        new_members = members[ass1 == i]

        new_gmm = {
            'mean': np.mean(new_members, axis=0),
            'cov': np.cov(new_members, rowvar=0) + np.identity(3)*1e-8,
            'size': orig_size * new_members.shape[0] / members.shape[0]
        }

        gmm_list.append(new_gmm)

    return gmm_list

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

def other_jaccard(segmentation, truth):
    intersection = np.int_((segmentation + truth) == 2)
    union = np.int_((segmentation + truth) != 0)

    return 100.0 * np.sum(intersection) / np.sum(union)

def jaccard_similarity(segmentation,truth):
    total = segmentation + truth
    temp = np.where(total == 2,1,total)
    temp = np.where(temp == 1,1,0)
    intersection = np.sum(np.sum(np.where(total == 2, 1, 0)))
    union = np.sum(np.sum(temp))

    return float(intersection)/float(union)*100

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        seg = grabcut(sys.argv[1])
        ground_truth = imread(sys.argv[2])
        ground_truth[ground_truth > 0] = 1
        accuracy = get_accuracy(seg, ground_truth)
        similarity = jaccard_similarity(seg, ground_truth)
        print 'OBTAINED FINAL ACCURACY: {}, JACCARD SIMILARITY: {}'.format(accuracy, similarity)
    else:
        test_grabcut()
