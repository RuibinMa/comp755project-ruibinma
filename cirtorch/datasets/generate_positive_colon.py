# this file should generate training data using colon images and colon SfM results.
import glob
import os
import argparse
import pickle
import sys
import logging
import random
import numpy as np
import PIL
from PIL import ImageStat
import shutil
sys.path.append('/playpen/software/pycolmap')
sys.path.append('/playpen/software/pycolmap/pycolmap')

from pycolmap import scene_manager
import datahelpers

sfm_results_dir_names = ['sfm_results_temp']
visualization_sample_n = 20

def is_hard_positive(image1, image2):
    point3D_ids1 = set(image1.point3D_ids)
    point3D_ids2 = set(image2.point3D_ids)
    overlap = len(point3D_ids1 & point3D_ids2)
    union = len(point3D_ids1) + len(point3D_ids2) - overlap
    ratio = overlap / union
    is_hard = False
    if ratio > 0:
        is_hard = True
    return is_hard, ratio


def process_trainval_cluster(clusterpath, global_results, phase='train'):
    images_dir = global_results['clusterpath_to_image_dir'][clusterpath]
    clusterid = global_results['clusterpath_to_clusterid'][clusterpath]
    scene = scene_manager.SceneManager(clusterpath)
    scene.load_images()
    # scene.load_points3D()
    image_ids = list(scene.images.keys())
    image_ids.sort()
    for _, image in scene.images.items():
        image_path = os.path.join(images_dir, image.name)
        if os.path.exists(image_path):
            global_results['image_path_to_idx'][image_path] = len(global_results['image_paths'])
            global_results['image_paths'].append(image_path)
            global_results['cluster'].append(clusterid)

            global_results[phase]['image_path_to_idx'][image_path] = len(global_results[phase]['image_paths'])
            global_results[phase]['image_paths'].append(image_path)
            global_results[phase]['cluster'].append(clusterid)
        else:
            logging.error('{} does not exist'.format(image_path))
    print('    {} images'.format(len(scene.images)))

    # find positive pairs
    for i in range(len(image_ids)):
        image1 = scene.images[image_ids[i]]
        for j in range(i+1, len(image_ids)):
            # judge whether pair <image_ids[i], image_ids[j]> is a hard positive pair.
            image2 = scene.images[image_ids[j]]
            is_hard, ratio = is_hard_positive(image1, image2)
            global_results['ratios'].append(ratio)
            if is_hard:
                image_path1 = os.path.join(images_dir, image1.name)
                image_path2 = os.path.join(images_dir, image2.name)
                image_global_id1 = global_results['image_path_to_idx'][image_path1]
                image_global_id2 = global_results['image_path_to_idx'][image_path2]
                # use image_global_id1 as query image
                global_results['qidxs'].append(image_global_id1)
                global_results['pidxs'].append(image_global_id2)
                # use image_global_id2 as query image
                global_results['qidxs'].append(image_global_id2)
                global_results['pidxs'].append(image_global_id1)

                image_phase_id1 = global_results[phase]['image_path_to_idx'][image_path1]
                image_phase_id2 = global_results[phase]['image_path_to_idx'][image_path2]
                global_results[phase]['qidxs'].append(image_phase_id1)
                global_results[phase]['pidxs'].append(image_phase_id2)
                global_results[phase]['qidxs'].append(image_phase_id2)
                global_results[phase]['pidxs'].append(image_phase_id1)

    return len(scene.images)

def process_test_cluster(clusterpath, global_results, num_samples=5):
    images_dir = global_results['clusterpath_to_image_dir'][clusterpath]
    scene = scene_manager.SceneManager(clusterpath)
    scene.load_images()
    num_images = len(scene.images)
    num_samples = min(num_images, num_samples)
    image_ids = scene.images.keys()
    for i in image_ids:
        image_path = os.path.join(images_dir, scene.images[i].name)
        global_results['test']['impath_to_id'][image_path] = len(global_results['test']['imlist'])
        global_results['test']['imlist'].append(image_path)
    query_image_ids = random.sample(image_ids, num_samples)
    for i in query_image_ids:
        query_image = scene.images[i]
        query_image_path = os.path.join(images_dir, scene.images[i].name)
        global_results['test']['qimlist'].append(query_image_path)
        global_results['test']['qidx'].append(global_results['test']['impath_to_id'][query_image_path])
        gnd = {'ok':[], 'junk':[]}
        for j in image_ids:
            compare_image = scene.images[j]
            compare_image_path = os.path.join(images_dir, compare_image.name)
            _, ratio = is_hard_positive(query_image, compare_image)
            # if ratio > 0 and ratio < 0.001:
            #     gnd['junk'].append(global_results['test']['impath_to_id'][compare_image_path])
            # elif ratio >= 0.001:
            #     gnd['ok'].append(global_results['test']['impath_to_id'][compare_image_path])
            if ratio > 0:
                gnd['ok'].append(global_results['test']['impath_to_id'][compare_image_path])
        global_results['test']['gnd'].append(gnd)

            
def get_num_images_of_cluster(clusterpath):
    scene = scene_manager.SceneManager(clusterpath)
    scene.load_images()
    return len(scene.images)


def get_clusters(case_dir, global_results, pick_largest=False):
    clusters = []
    images_dir = os.path.join(case_dir, 'images')
    total_num_clusters = 0
    if pick_largest:
        # in this case, only pick the largest cluster from each set of keyframes
        sfm_results_dir = os.path.join(case_dir, 'sfm_results_temp')
        clusters = []
        keyframe_sets = os.listdir(sfm_results_dir)
        for keyframe_set in keyframe_sets:
            sub_clusters = os.listdir(os.path.join(sfm_results_dir, keyframe_set))
            total_num_clusters += len(sub_clusters)
            max_num_images = 0
            largest_cluster = None
            for sub_cluster in sub_clusters:
                sub_cluster_path = os.path.join(sfm_results_dir, keyframe_set, sub_cluster)
                num_images = get_num_images_of_cluster(sub_cluster_path)
                if num_images > max_num_images:
                    max_num_images = num_images
                    largest_cluster = sub_cluster_path
            if largest_cluster:
                clusters.append(largest_cluster)
    else:
        # in this case, pick all SfM clusters
        sfm_results_dir = os.path.join(case_dir, 'sfm_results')
        clusters = [os.path.join(sfm_results_dir, c) for c in os.listdir(sfm_results_dir)]
        total_num_clusters += len(clusters)

    for clusterpath in clusters:
        global_results['clusterpaths'].append(clusterpath)
        global_results['clusterpath_to_image_dir'][clusterpath] = images_dir
    print('picked [{}/{}] clusters in case {}'.format(len(clusters), total_num_clusters, case_dir))


def shuffle(global_results):
    # shuffle the pairs of query and positive
    print('Shuffling queries and positives')
    if len(global_results['qidxs']) == 0:
        return
    combined = list(zip(global_results['qidxs'], global_results['pidxs']))
    random.shuffle(combined)
    global_results['qidxs'], global_results['pidxs'] = zip(*combined)


def visualize(global_results, visualization_dir, sample_n=50):
    if os.path.exists(visualization_dir):
        shutil.rmtree(visualization_dir)
    os.makedirs(visualization_dir)
    for phase in ['train', 'val']:
        phase_dir = os.path.join(visualization_dir, phase)
        os.makedirs(phase_dir)
        for i in range(min(len(global_results[phase]['image_paths']), sample_n)):
            image_path1 = global_results[phase]['image_paths'][global_results[phase]['qidxs'][i]]
            image_path2 = global_results[phase]['image_paths'][global_results[phase]['pidxs'][i]]

            image1 = datahelpers.pil_loader(image_path1)
            image2 = datahelpers.pil_loader(image_path2)
            width, height = image1.size
            combined = PIL.Image.new('RGB', (width*2, height))
            combined.paste(image1, (0, 0))
            combined.paste(image2, (width, 0))
            combined.save(os.path.join(phase_dir, 'combined{:03d}.jpg'.format(i)))


def visualize_test(global_results, visualization_dir):
    if os.path.exists(visualization_dir):
        shutil.rmtree(visualization_dir)
    os.makedirs(visualization_dir)
    for i, query_image_path in enumerate(global_results['test']['qimlist']):

        query_image = datahelpers.pil_loader(query_image_path)
        width, height = query_image.size

        if global_results['test']['gnd'][i]['ok']:
            ok_image_id = random.choice(global_results['test']['gnd'][i]['ok'])
            ok_image_path = global_results['test']['imlist'][ok_image_id]
            ok_image = datahelpers.pil_loader(ok_image_path)
        else:
            ok_image = PIL.Image.new('RGB', (width, height))

        if global_results['test']['gnd'][i]['junk']:
            junk_image_id = random.choice(global_results['test']['gnd'][i]['junk'])
            junk_image_path = global_results['test']['imlist'][junk_image_id]
            junk_image = datahelpers.pil_loader(junk_image_path)
        else:
            junk_image = PIL.Image.new('RGB', (width, height))

        combined = PIL.Image.new('RGB', (width*3, height))
        combined.paste(query_image, (0, 0))
        combined.paste(ok_image, (width, 0))
        combined.paste(junk_image, (width*2, 0))
        combined.save(os.path.join(visualization_dir, 'combined{:03d}.jpg'.format(i)))


def generate(base_dir, output_dir, pick_largest=False, percentage_test=0.1, percentage_val=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    colonoscopy_cases = glob.glob(os.path.join(base_dir, 'Auto_*'))
    colonoscopy_cases.sort()

    global_results = {}
    global_results['clusterpaths'] = []
    global_results['clusterpath_to_clusterid'] = {}
    global_results['clusterpath_to_image_dir'] = {}
    global_results['image_paths'] = []
    global_results['image_path_to_idx'] = {}
    global_results['cluster'] = []
    global_results['qidxs'] = []
    global_results['pidxs'] = []
    global_results['ratios'] = []
    global_results['train'] = {'qidxs':[], 'pidxs':[], 'image_paths':[], 'cluster':[], 'image_path_to_idx':{}}
    global_results['val'] = {'qidxs':[], 'pidxs':[], 'image_paths':[], 'cluster':[], 'image_path_to_idx':{}}
    global_results['test'] = {'imlist':[], 'qimlist':[], 'gnd':[], 'impath_to_id':{}, 'qidx':[]}

    # get all clusters into global_results
    for case in colonoscopy_cases:
        # missing case
        if not case.endswith('Auto_A_Aug18_08-10-41'):
            get_clusters(case, global_results, pick_largest=pick_largest)
    # process the clusters
    # the clusters will be divided into two partitions, one for generating train and val, one for test
    num_test_clusters = int(percentage_test * len(global_results['clusterpaths']))
    num_val_clusters = int(percentage_val * len(global_results['clusterpaths']))
    num_train_clusters = len(global_results['clusterpaths']) - num_test_clusters - num_val_clusters
    random.shuffle(global_results['clusterpaths'])
    for i, clusterpath in enumerate(global_results['clusterpaths']):
        global_results['clusterpath_to_clusterid'][clusterpath] = i
    print('clusters shuffled.')
    # process for train
    for i in range(num_train_clusters):
        clusterpath = global_results['clusterpaths'][i]
        print('Train [{}/{}] processing cluster {}'.format(i+1, num_train_clusters, clusterpath), end='')
        process_trainval_cluster(clusterpath, global_results, phase='train')
    
    # process for val
    for i in range(num_val_clusters):
        clusterpath = global_results['clusterpaths'][i+num_train_clusters]
        print('Val   [{}/{}] processing cluster {}'.format(i+1, num_val_clusters, clusterpath), end='')
        process_trainval_cluster(clusterpath, global_results, phase='val')

    assert(len(global_results['image_paths'])==len(global_results['cluster']))
    assert(len(global_results['pidxs'])==len(global_results['qidxs']))
    assert(len(global_results['train']['image_paths'])==len(global_results['train']['cluster']))
    assert(len(global_results['train']['pidxs'])==len(global_results['train']['qidxs']))
    assert(len(global_results['val']['image_paths'])==len(global_results['val']['cluster']))
    assert(len(global_results['val']['pidxs'])==len(global_results['val']['qidxs']))
    print('Total number of train/val images: {}'.format(len(global_results['image_paths'])))
    print('Total number of train/val pairs:  {}'.format(len(global_results['qidxs'])))
    print('Number of train images:{}'.format(len(global_results['train']['image_paths'])))
    print('Number of train pairs: {}'.format(len(global_results['train']['qidxs'])))
    print('Number of val   images:{}'.format(len(global_results['val']['image_paths'])))
    print('Number of val   pairs: {}'.format(len(global_results['val']['qidxs'])))
    print('Percentiles:')
    print(np.percentile(global_results['ratios'], [10,20,30,40,50,60,70,80,90,100]))
    # shuffle
    shuffle(global_results)
    shuffle(global_results['train'])
    shuffle(global_results['val'])
    
    # process for test
    for i in range(num_test_clusters):
        clusterpath = global_results['clusterpaths'][i+num_train_clusters+num_val_clusters]
        print('Test  [{}/{}] processing cluster {}'.format(i+1, num_test_clusters, clusterpath))
        process_test_cluster(clusterpath, global_results)
    assert(len(global_results['test']['qimlist'])==len(global_results['test']['qidx']))
    assert(len(global_results['test']['qimlist'])==len(global_results['test']['gnd']))
    print('Total number of test images: {}'.format(len(global_results['test']['imlist'])))
    print('Total number of query images: {}'.format(len(global_results['test']['qimlist'])))

    # save to .pkl files
    with open(os.path.join(output_dir, 'colon.pkl'), 'wb') as handle:
        pickle.dump(global_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_dir, 'gnd_colon.pkl'), 'wb') as handle:
        pickle.dump(global_results['test'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    visualization_trainval_dir = os.path.join(output_dir, 'visualization_trainval')
    visualize(global_results, visualization_trainval_dir)
    print('Visualization train/val image samples written to {}'.format(visualization_trainval_dir))
    visualization_test_dir = os.path.join(output_dir, 'visualization_test')
    visualize_test(global_results, visualization_test_dir)
    print('Visualization test image samples written to {}'.format(visualization_test_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create training data for cnnimageretrieval-pytorch.')
    parser.add_argument('--output_dir', '-o', help='output directory',
                        default='/playpen/cnnimageretrieval-pytorch/data/colon')
    parser.add_argument('--pick_largest', help='whether only pick the largest cluster in each keyframe set.', action='store_true')
    args = parser.parse_args()
    generate('/media/ruibinma/My Passport/playpen/colondata', args.output_dir, args.pick_largest, percentage_test=0.1, percentage_val=0.0)
