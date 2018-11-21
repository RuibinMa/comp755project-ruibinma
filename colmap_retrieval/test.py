import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import pickle
import shutil
import subprocess
from cirtorch.utils.evaluate import compute_map

def get_scores(retrieval_result_path, cfg):
    print('getting score from on '.format(retrieval_result_path))
    num_images = len(cfg['imlist'])
    num_queries = len(cfg['qidx'])
    scores = np.zeros(shape=(num_images, num_queries))
    count = 0
    with open(retrieval_result_path) as result_file:
        image_i = -1
        for line in result_file:
            if line.startswith("Indexing"):
                continue
            if line.startswith("Querying"):
                strs = line.split(" ")
                image_name = strs[3]
                image_i = cfg['name_to_qid'][image_name]
                # print("Query image: {} [{}]".format(image_name, image_i))
                count += 1
            else:
                _, image_name, score = line.split(",")
                image_name = image_name.split('=')[1]
                image_j = cfg['name_to_id'][image_name]
                score = float(score.split("=")[1])
                scores[image_j][image_i] = score
            
            if count > num_queries:
                raise AssertionError('number of queries in report is larger than in groundtruth, impossible')
    return scores


def run_colmap_retriever(cfg):
    tmp_dir = '/tmp/colmap_retriever_test'
    image_dir = os.path.join(tmp_dir, 'images')
    query_image_list_path = os.path.join(tmp_dir, 'query_image_list.txt')
    retrieval_result_path = os.path.join(tmp_dir, 'colmap_retrieval_result.txt')
    if os.path.exists('/tmp/colmap_retriever_test'):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    os.makedirs(image_dir)
    name_to_id = {}
    cfg['unique_names'] = []
    for i, image_path in enumerate(cfg['imlist']):
        image_name = os.path.basename(image_path)
        extension = os.path.splitext(image_name)[1]
        unique_name = 'image{:010d}{}'.format(i, extension)
        shutil.copyfile(image_path, os.path.join(image_dir, unique_name))
        name_to_id[unique_name] = i
        cfg['unique_names'].append(unique_name)
    cfg['name_to_id'] = name_to_id
    # feature_extractor
    cmd = ['colmap', 'feature_extractor',
           '--database_path', os.path.join(tmp_dir, 'database.db'),
           '--image_path', image_dir,
           # '--ImageReader.camera_model', 'PINHOLE',
           '--ImageReader.camera_model', 'SIMPLE_RADIAL_FISHEYE',
           '--ImageReader.single_camera', '1',
           #'--ImageReader.camera_params', '374.1583,375.3094,333.3978,273.8418',
           '--SiftExtraction.domain_size_pooling', '1']
    print(cmd)
    subprocess.call(cmd)

    # vocab_tree_retriever
    print('running vocab_tree_retriever...')
    cfg['name_to_qid'] = {}
    with open(query_image_list_path, 'w') as file:
        for i, q in enumerate(cfg['qidx']):
            # verify
            if 'qimlist' in cfg:
                assert(cfg['qimlist'][i] == cfg['imlist'][q])
            file.write(cfg['unique_names'][q]+'\n')
            cfg['name_to_qid'][cfg['unique_names'][q]] = i
    cmd = ['colmap', 'vocab_tree_retriever',
           '--database_path', os.path.join(tmp_dir, 'database.db'),
           '--query_image_list_path', query_image_list_path,
           '--vocab_tree_path', '/playpen/vocab_tree/vocab_tree-262144.bin',
           '--num_images_after_verification', str(len(cfg['imlist']))]
    with open(retrieval_result_path, 'w') as file:
        subprocess.call(cmd, stdout=file)
    return retrieval_result_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test colmap retrieval performance')
    parser.add_argument('ground_truth', help='Path to .pkl which stores the groundtruth data of test images.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--load-score-path', help='load score matrix from this path (numpy, .bin)', default=None)
    group.add_argument('--save-score-path', help='save score matrix into this path (numpy, .bin)', default=None)
    parser.add_argument('--result_path', help='Save result (mAP, mP@k) into this path (.pkl)')
    args = parser.parse_args()

    ground_truth = args.ground_truth
    with open(ground_truth, 'rb') as file:
        cfg = pickle.load(file)
    
    # load scores directly from a previous result
    if args.load_score_path:
        with open(args.load_score_path, 'rb') as file:
            scores = np.load(file)
    else:
        # run vocab_tree_retriever to get the scores
        retrieval_result_path = run_colmap_retriever(cfg)
        scores = get_scores(retrieval_result_path, cfg)
        # save scores matrix if specified
        if args.save_score_path:
            with open(args.save_score_path, 'wb') as file:
                np.save(file, scores)

    # get ranks matrix which is of the same size as scores
    # kappas = [1, 5, 10, 50, 100, 500, 1000, 5000]
    kappas = np.arange(len(scores)) + 1
    ranks = np.argsort(-scores, axis=0)
    mAP, aps, pr, prs = compute_map(ranks, cfg['gnd'], kappas=kappas)
    print('mean average precision: {:.2f}'.format(mAP*100.))
    # print('mP@k{}: {}'.format(kappas, np.around(pr*100, decimals=2)))

    if args.result_path:
        with open(args.result_path, 'wb') as file:
            result = {}
            result['scores'] = scores
            result['map'] = mAP
            result['aps'] = aps
            result['pr'] = pr
            result['prs'] = prs
            result['kappas'] = kappas
            result['ranks'] = ranks

            pickle.dump(result, file)