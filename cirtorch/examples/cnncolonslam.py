import numpy as np
import pickle
import argparse
import os
import time
from PIL import Image
import random
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors, extract_ss, extract_ms
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime
from cirtorch.datasets.genericdataset import ImagesFromList

class Node(object):
    def __init__(self):
        self.observations = []
    def add_observation(self, observation):
        self.observations.append(observation)
    @property
    def num_observations(self):
        return len(observations)

class Graph(object):

    SAMPLE_SIZE = 2000
    SIGMA = 5

    def __init__(self, unobserved_sample_observations):
        self.nodes = []
        self.Lnew = 0
        self.unobserved_sample_observations = unobserved_sample_observations

    def add_node(self, node):
        if not isinstance(node, Node):
            raise AttributeError('The node to be inserted must be a Node instance')
        self.nodes.append(node)
        self.Lnew = len(self.nodes)

    def likelihoods(self, observation):
        res = np.zeros(self.num_observed_nodes + 1)
        for i in range(self.num_observed_nodes):
            res[i] = np.dot(self.nodes[i].observations[0], observation)
        return res
    
    def priors_FABMAP(self, last_location):
        res = np.zeros(self.num_observed_nodes + 1)
        if last_location == None:
            res[0] = 1
        elif last_location == self.num_observed_nodes-1:
            res[-1] = 0.9
            res[:-1] = (1-res[-1])/(len(res)-1)
        elif last_location == 0:
            res[:2] = 0.5
        else:
            res[last_location-1:last_location+2] = 1./3.
        return res
    
    def priors(self, last_location):
        res = np.zeros(self.num_observed_nodes + 1)
        if last_location == None:
            res[0] = 1
        elif last_location == 0:
            res[:2] = [0.1, 0.9]
        else:
            res[last_location-1:last_location+2] = [0.05, 0.05, 0.9]
        return res
    
    def unobserved_marginal(self, observation, unobserved_prior):
        samples = np.random.choice(self.num_unobserved_samples, min(self.SAMPLE_SIZE, self.num_unobserved_samples), replace=False)
        sample_descriptors = self.unobserved_sample_observations[samples, :]
        likelihoods = np.matmul(sample_descriptors, observation)
        return np.sum(likelihoods) * unobserved_prior / float(self.num_unobserved_samples)

    def determine_location(self, observation, last_location):
        if last_location and (last_location >= self.num_observed_nodes or last_location < 0):
            raise ValueError('Last location {} must be None or in ranges [0,{}]'.format(last_location, self.num_observed_nodes-1))
        
        likelihoods = self.likelihoods(observation)
        priors = self.priors(last_location)
        observed_marginal = np.sum(likelihoods[:self.num_observed_nodes] * priors[:self.num_observed_nodes])
        unobserved_marginal = self.unobserved_marginal(observation, priors[-1])
        marginal = observed_marginal + unobserved_marginal

        posteriors = likelihoods * priors / marginal
        posteriors[-1] = 1 - np.sum(posteriors[:-1])
        assert(posteriors[-1] >= 0)
        location = np.argmax(posteriors)
        probability = posteriors[location]
        print(likelihoods)
        print(priors)
        print(observed_marginal, unobserved_marginal, marginal)
        print(posteriors)
        return location, probability
        
    
    @property
    def num_observed_nodes(self):
        return len(self.nodes)
    @property
    def num_unobserved_samples(self):
        return len(self.unobserved_sample_observations)
    

def image_loader(input_dir, image_size, transform, bbxs=None):
    image_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir)]
    image_paths.sort()
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=image_paths, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )
    return loader

def colon_slam(net, loader, unobserved_descriptors, ms=[1], msp=1):

    # vecs = torch.zeros(net.meta['outputdim'], len(images))
    graph = Graph(unobserved_descriptors)
    location = None
    for i, input in enumerate(loader):
        print('Frame [{}]    '.format(i))
        input_var = Variable(input.cuda())

        if len(ms) == 1:
            descriptor = extract_ss(net, input_var)
        else:
            descriptor = extract_ms(net, input_var, ms, msp)
        descriptor = descriptor.numpy()

        location, probability = graph.determine_location(descriptor, location)
        if location >= graph.num_observed_nodes:
            new_node = Node()
            new_node.add_observation(descriptor)
            graph.add_node(new_node)
        
        print("Location={}({:.2f})".format(location, probability*100.))
def visualization(net, loader, ms, msp, num_images):
    des_vis = np.empty(shape=(num_images, 2048))
    for i, input in enumerate(loader):
        print('Frame [{}]    '.format(i))
        input_var = Variable(input.cuda())

        if len(ms) == 1:
            descriptor = extract_ss(net, input_var)
        else:
            descriptor = extract_ms(net, input_var, ms, msp)
        descriptor = descriptor.numpy()
        des_vis[i, :] = descriptor
    
    embedding = TSNE(n_components=2).fit_transform(des_vis)
    plt.plot(embedding[:,0], embedding[:, 1], '.')
    plt.plot(embedding[0,0], embedding[0, 1], 'ro')
    plt.plot(embedding[-1,0],embedding[-1,1], 'yo')
    plt.savefig('/home/ruibinma/Desktop/vis_seq.png')

def main(args):
    input_dir = args.input
    images = os.listdir(input_dir)
    print('{} images'.format(len(images)))

    if args.network_path is not None:
        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        state = torch.load(args.network_path)
        net = init_network(model=state['meta']['architecture'], pooling=state['meta']['pooling'], whitening=state['meta']['whitening'], 
                           mean=state['meta']['mean'], std=state['meta']['std'], pretrained=False)
        net.load_state_dict(state['state_dict'])
        
        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = [1]
    msp = 1
    if args.multiscale:
        ms = [1, 1./math.sqrt(2), 1./2]
        if net.meta['pooling'] == 'gem' and net.whiten is None:
            msp = net.pool.p.data.tolist()[0]
    
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    loader = image_loader(input_dir, args.image_size, transform)

    # extract unobserved sample descriptors

    if args.load_unobserved_observation:
        with open(args.load_unobserved_observation, 'rb') as file:
            unobserved_descriptors = np.load(file)
    else:
        with open(args.unobserved_data, 'rb') as file:
            unobserved_images = pickle.load(file)['image_paths']
        unobserved_descriptors = extract_vectors(net, unobserved_images, args.image_size, transform, ms=ms, msp=msp)
        unobserved_descriptors = unobserved_descriptors.numpy().T

    if args.save_unobserved_observation:
        with open(args.save_unobserved_observation, 'wb') as file:
            np.save(file, unobserved_descriptors)
    # 1D colon SLAM system
    #X_embedded = TSNE(n_components=2).fit_transform(unobserved_descriptors)
    #plt.plot(X_embedded[:,0], X_embedded[:, 1], '.')
    #plt.savefig('/home/ruibinma/Desktop/vis.png')
    visualization(net, loader, ms, msp, len(images))
    #colon_slam(net, loader, unobserved_descriptors, ms, msp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='1D SLAM system for colonoscopy')
    parser.add_argument('--input', '-i', help='Input image sequence folder.')
    parser.add_argument('--network-path', '-n', help='Path to the cnn network to compute image descriptors (.pth)')
    parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                        help='maximum size of longer image side used for testing (default: 1024)')
    parser.add_argument('--multiscale', '-ms', dest='multiscale', action='store_true',
                        help='use multiscale vectors for testing')
    parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None)
    parser.add_argument('--gpu-id', default='0', help='gpu id used for testing (default=0)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--unobserved-data', '-u', help='Path to the database of unobserved samples (.pkl)')
    group.add_argument('--load-unobserved-observation', '-l',
                        help='If specified, load nxp matrix from this path as the unobserved descriptors')

    parser.add_argument('--save-unobserved-observation', '-s',
                        help='If specified, save nxp matrix of the unobserved descriptors to this parh.')
    args = parser.parse_args()

    main(args)