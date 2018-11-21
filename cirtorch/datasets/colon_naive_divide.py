# For each case, first uniformly sample 1/3 images
# Divide these images into 200 image groups
# Uniformly keep 1/2 groups, in order to separate the groups
# A typical case contains 12,000 images -> 4000 images -> 2000 images (10 keyframe sets)
# 2000 * 19 = 38000 training images
# 

import argparse
import os
import glob
import shutil

def split_images(case_dir, image_sample_rate, group_size, group_sample_rate):
    print('Splitting {}'.format(case_dir))
    image_dir = os.path.join(case_dir, 'images')
    keyframe_dir = os.path.join(case_dir, 'naive')
    if os.path.exists(keyframe_dir):
        shutil.rmtree(keyframe_dir)
    os.makedirs(keyframe_dir)

    image_names = os.listdir(image_dir)
    image_names.sort()
    # uniformly sample images
    image_names = image_names[::image_sample_rate]
    # divide into groups
    i = 0
    count = 0
    while(i * group_size < len(image_names)):
        if i % group_sample_rate == 0:
            group = image_names[i*group_size: (i+1)*group_size]
            group_dir = os.path.join(keyframe_dir, '{:02d}'.format(count))
            os.makedirs(group_dir)
            for image_name in group:
                source_name = os.path.join(image_dir, image_name)
                target_name = os.path.join(group_dir, image_name)
                shutil.copyfile(source_name, target_name)
            count += 1
        i += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Colon data folder')
    parser.add_argument('--image_sample_rate', type=int, default=3)
    parser.add_argument('--group_sample_rate', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=200)
    args = parser.parse_args()

    case_dirs = glob.glob(os.path.join(args.data_dir, 'Auto*'))
    for case_dir in case_dirs:
        split_images(case_dir, args.image_sample_rate, args.group_size, args.group_sample_rate)
    
