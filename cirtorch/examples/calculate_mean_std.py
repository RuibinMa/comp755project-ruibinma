import os
from PIL import Image
import numpy as np
import argparse
import pickle

def compute(data):
    img = Image.open(data['image_paths'][0]).convert('RGB')
    acc_img = np.zeros_like(np.array(img), dtype=float)
    print('Image shape = {}'.format(acc_img.shape))

    # calculate mean
    for i, img_path in enumerate(data['image_paths']):
        img = Image.open(img_path).convert('RGB')
        acc_img += np.array(img) / 255.0
        if i % 1000 == 0:
            print('{}/{}'.format(i, len(data['image_paths'])))
    acc_img /= len(data['image_paths'])
    mean = np.mean(acc_img, axis=(0,1))

    # calculate std
    std_img = np.zeros_like(acc_img, dtype=float)
    for i, img_path in enumerate(data['image_paths']):
        img = np.array(Image.open(img_path).convert('RGB')) / 255.0
        std_img[:,:,0] += (img[:,:,0] - mean[0])**2
        std_img[:,:,1] += (img[:,:,1] - mean[1])**2
        std_img[:,:,2] += (img[:,:,2] - mean[2])**2
        if i % 1000 == 0:
            print('{}/{}'.format(i, len(data['image_paths'])))
    std_img /= len(data['image_paths'])
    std = np.sqrt(np.mean(std_img, axis=(0,1)))        
    return mean, std
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate mean and std for training data.')
    parser.add_argument('train_data', help='Path to training data, which is a .pkl of a folder')
    args = parser.parse_args()

    if args.train_data.endswith('.pkl'):
        with open(args.train_data, 'rb') as file:
            data = pickle.load(file)        
            if 'train' in data:
                arg_data = data['train']
            else:
                arg_data = data
    else:
        arg_data = {}
        image_paths = [os.path.join(args.train_data, imname) for imname in os.listdir(args.train_data)]
        image_paths.sort()
        arg_data['image_paths'] = image_paths

    mean, std = compute(arg_data)
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    if 'meta' not in arg_data:
        arg_data['meta'] = {}
    arg_data['meta']['mean'] = mean
    arg_data['meta']['std'] = std

    # write result back into the training data
    if args.train_data.endswith('.pkl'):
        with open(args.train_data, 'wb') as file:
            pickle.dump(data, file)
    

