import os
import pickle
import argparse
import glob
import numpy as np
import shutil

from PIL import Image
import random

import torch
from torch.autograd import Variable
from torchvision import transforms
from cirtorch.networks.imageretrievalnet import init_network, extract_ss, extract_ms
from cirtorch.datasets.genericdataset import ImagesFromList


def generate_overlap(images, params, vecs, results):
    count = 0
    for i, im1 in enumerate(images):
        sample = [i]
        for j in range(i+params['min_gap'], i+params['window_size']):
            if len(sample) == params['n']:
                break
            if j >= len(images):
                break
            if j < sample[-1] + params['min_gap']:
                continue
            similarity = np.dot(vecs[sample[-1], :], vecs[j, :])
            if similarity > 0.85 and similarity <0.95:
                sample.append(j)
        
        if len(sample) == params['n']:
            results['samples'].append([images[i] for i in sample])
            count += 1
        if (i+1)%10 == 0 or i+1 == len(images):
            print('\r>>>> {}/{} [{}] found...'.format((i+1), len(images), count), end='')
    print('')
        


def generate_non_overlap(images, min_gap, vecs, results):
    pass


def generate(image_dir, n, net, net_params, results, descriptors_path=None):
    print('Generating samples from {} [{}-tuple]'.format(image_dir, n))
    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    # first, extract descriptors for measuring similarity
    if descriptors_path and os.path.exists(descriptors_path):
        with open(descriptors_path, 'rb') as file:
            vecs = np.load(file)
        print('Loaded descriptors [{}x{}]'.format(vecs.shape[0], vecs.shape[1]))
    else:
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=images, imsize=362, bbxs=None, transform=net_params['transform']),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        vecs = torch.zeros(len(images), net.meta['outputdim'])
        print('Extracting descriptors [{}x{}]...'.format(vecs.shape[0], vecs.shape[1]))
        for i, input in enumerate(loader):
            input_var = Variable(input.cuda())
            if len(net_params['ms']) == 1:
                vecs[i, :] = extract_ss(net, input_var)
            else:
                vecs[i, :] = extract_ms(net, input_var, net_params['ms'], net_params['msp'])
            
            if (i+1)%10 == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        vecs = vecs.numpy()
        print('')
        if descriptors_path:
            with open(descriptors_path, 'wb') as file:
                np.save(file, vecs)

    # second, sequentially extract training samples
    params = {}
    params['n'] = n
    params['min_gap'] = 3 # frames
    params['window_size'] = 300 # frames
    params['min_similarity'] = 0.9
    overlap = True
    if overlap:
        generate_overlap(images, params, vecs, results)
    else:
        generate_non_overlap(images, min_gap, vecs, results)


def visualization(results, num_vis, vis_dir):
    vis_samples = random.sample(results['samples'], num_vis)
    for j, sample in enumerate(vis_samples):
        n = len(sample)
        im = Image.open(sample[0])
        width, height = im.size
        combined = Image.new('RGB', (width*n, height))
        
        for i, impath in enumerate(sample):
            im = Image.open(impath)
            combined.paste(im, (i*width, 0))
        combined.save(os.path.join(vis_dir, 'sample{:02d}.jpg'.format(j)))


def main(args):
    # prepare the retrival network
    state = torch.load(args.network_path)
    net = init_network(
        model=state['meta']['architecture'], pooling=state['meta']['pooling'],
        whitening=state['meta']['whitening'], mean=state['meta']['mean'],
        std=state['meta']['std'], pretrained=False
    )
    net.load_state_dict(state['state_dict'])
    if 'Lw' in state['meta']:
        net.meta['Lw'] = state['meta']['Lw']
    
    print(">>>> loaded network: ")
    print(net.meta_repr())
    
    ms = [1]
    msp = 1
    if args.multiscale:
        ms = [1, 1./math.sqrt(2), 1./2]
        if net.meta['pooling'] == 'gem' and net.whiten is None:
            msp = net.pool.p.data.tolist()[0]
    
    net.cuda()
    net.eval()
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    net_params = {}
    net_params['ms'] = ms
    net_params['msp'] = msp
    net_params['transform'] = transform

    # processing case by case
    if args.descriptors_dir and not os.path.exists(args.descriptors_dir):
        os.makedirs(args.descriptors_dir)

    cases = os.listdir(args.base_dir)
    results = {}
    results['samples'] = []
    for case in cases[:5]:
        image_dir = os.path.join(args.base_dir, case, 'images')
        descriptors_path = None
        if args.descriptors_dir:
            descriptors_path = os.path.join(args.descriptors_dir, case+'.bin')
        generate(image_dir, args.n, net, net_params, results, descriptors_path)
    
    # visualization
    if args.vis_dir:
        if os.path.exists(args.vis_dir):
            shutil.rmtree(args.vis_dir)
        os.makedirs(args.vis_dir)
        num_vis = min(100, len(results['samples']))
        visualization(results, num_vis, args.vis_dir)


    # store result
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, 'RNN_traindata.pkl')
    print('{} samples generated.'.format(len(results['samples'])))
    print('Writing to {}'.format(output_path))
    with open(output_path, 'wb') as file:
        pickle.dump(results, file)
    print('Done.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Colon RNN training data generator')
    parser.add_argument('--base_dir', '-b', help='Base folder to colon images')
    parser.add_argument('--n', '-n', type=int, help='Each training sample should contain this number of images')
    parser.add_argument('--output_dir', '-o', help='Output directory')
    parser.add_argument('--gpu_id', default='0', help='gpu id used for testing (default: 0)')
    parser.add_argument('--network_path', help='network path, destination where network is saved')
    parser.add_argument('--multiscale', '-ms', dest='multiscale', action='store_true', help='use multiscale vectors for testing')
    parser.add_argument('--vis_dir', help='Visualization Directory')
    parser.add_argument('--descriptors_dir', help='Directory to write or load descriptors of each case. This is like a cache.')
    args = parser.parse_args()
    main(args)