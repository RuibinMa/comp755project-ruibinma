'''
Created on Oct 16, 2017

@author: ruibinma
'''
from scene_manager import SceneManager
def cvt_bin_txt(model_folder):
    converter = SceneManager( model_folder )
    converter.load_cameras()
    converter.load_images()
    converter.load_points3D()
    
    converter.save_cameras(output_folder=model_folder, output_file=None, binary=False)
    converter.save_images(output_folder=model_folder, output_file=None, binary=False)
    converter.save_points3D(output_folder=model_folder, output_file=None, binary=False)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_folder", default="sfm_results/0")
    args = parser.parse_args()
    
    cvt_bin_txt(args.model_folder)
