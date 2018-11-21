import os
import pickle
import argparse
import shutil
import random
from PIL import Image

num_vis = 100

parser = argparse.ArgumentParser('Modify the names in colon.pkl, gnd_xxx.pkl')
parser.add_argument('--train_data', help='Path to training data.')
parser.add_argument('--test_data', help='Path to testing data.')
parser.add_argument('--new_base', help='Path to the new base dir.')
parser.add_argument('--imfolder', help='folder in each case that store images.', default='images-raw')
parser.add_argument('--img_suffix', help='jpg, png, etc', default='jpg')
parser.add_argument('--vis', help='path to visualization folder')

args = parser.parse_args()

train_path = args.train_data
test_path = args.test_data
vis_dir = args.vis

def change_path(data, new_base, imfolder, img_suffix, vis_dir, phase):
    samples = random.sample(range(len(data)), min(len(data), num_vis))
    for i, impath in enumerate(data):
        parts = impath.split('/')
        imname = parts[-1]
        imname = parts[-1].split('.')[0] + '.' + img_suffix
        case = parts[-3] + '.mov'
        new_im_dir = os.path.join(new_base, case, imfolder)
        newpath = os.path.join(new_base, case, imfolder, imname)
        if not os.path.exists(new_im_dir):
            os.makedirs(new_im_dir)
        shutil.copyfile(impath, newpath)

        if not os.path.exists(newpath):
            print('{} does not exist'.format(newpath))
        data[i] = newpath
        if i in samples:
            im = Image.open(impath)
            im2 = Image.open(newpath)
            width, height = im.size
            combined = Image.new('RGB', (width*2, height))
            combined.paste(im, (0, 0))
            combined.paste(im2, (width, 0))
            combined.save(os.path.join(vis_dir, '{}{:03d}.jpg'.format(phase, i)))


def change_dict(data, key, new_base, imfolder, img_suffix):
    new_dict = {}
    for impath, id in data[key].items():
        parts = impath.split('/')
        imname = parts[-1]
        imname = parts[-1].split('.')[0] + '.' + img_suffix
        case = parts[-3] + '.mov'
        newpath = os.path.join(new_base, case, imfolder, imname)
        new_dict[impath] = id
    data[key] = new_dict

if vis_dir and os.path.exists(vis_dir):
    shutil.rmtree(vis_dir)
os.makedirs(vis_dir)

with open(train_path, 'rb') as file:
    train_data = pickle.load(file)
with open(test_path, 'rb') as file:
    test_data = pickle.load(file)

change_path(train_data['image_paths'], args.new_base, args.imfolder, args.img_suffix, vis_dir, 'whole')
change_path(train_data['train']['image_paths'], args.new_base, args.imfolder, args.img_suffix, vis_dir, 'train')
change_dict(train_data['train'], 'image_path_to_idx',  args.new_base, args.imfolder, args.img_suffix)
change_path(train_data['val']['image_paths'], args.new_base, args.imfolder, args.img_suffix, vis_dir, 'val')
change_dict(train_data['val'], 'image_path_to_idx',  args.new_base, args.imfolder, args.img_suffix)

change_path(test_data['imlist'], args.new_base, args.imfolder, args.img_suffix, vis_dir, 'test')
change_path(test_data['qimlist'], args.new_base, args.imfolder, args.img_suffix, vis_dir, 'query')
change_dict(test_data, 'impath_to_id', args.new_base, args.imfolder, args.img_suffix)

train_parts = train_path.split('/')
train_parts[-1] = 'new_' + train_parts[-1]
new_train_path = '/'.join(train_parts)

test_parts = test_path.split('/')
test_parts[-1] = 'new_' + test_parts[-1]
new_test_path = '/'.join(test_parts)

with open(new_train_path, 'wb') as file:
    pickle.dump(train_data, file, protocol=pickle.HIGHEST_PROTOCOL)
with open(new_test_path, 'wb') as file:
    pickle.dump(test_data, file, protocol=pickle.HIGHEST_PROTOCOL)