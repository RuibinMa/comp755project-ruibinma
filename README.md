# Colon Place Recognition Using Convolutional Neural Network

This project trains a CNN for place recognition (image retrieval) task in colonoscopic images.

All the commands should be excuted under the top-level directory (comp755project-ruibinma/)

First, combine the sharded compressed files:
```
# Due to the size limit of github, the uploaded model and data are sharded
cat pretrained_model/model.pth.tar.parta* > pretrained_model/model.pth.tar
cat data/colonimages.tar.gz.parta* > data/colonimages.tar.gz
# Extract colon images
cd data
tar -xzvf colonimages.tar.gz
cd ..
```

There are three folders of code in the home directory:
```
### ./cirtorch/
contains training and testing functions of CNN image retrieval (the main part of this project)
### ./colmap_retrieval/
testing function of a SIFT-based vocabulary-tree retriever. (for comparison with CNN)
### ./pycolmap/
a third-party software that contains APIs used to generate training samples.
```

To test the pretrained model on colon images and get mean average precision:
```
python3 -m cirtorch.examples.test -npath pretrained_model/model_epoch28.pth.tar --datasets 'colon'
```
To train the CNN image retriever using colon images
```
python3 -m cirtorch.examples.train_colon EXPORT_DIR --training-dataset 'colon' --test-dataset 'colon' --arch 'resnet50' --pool 'gem' --loss 'contrastive' --pool-size=20000 --batch-size 5 --image-size 362
```
To test the vocabulary tree image retriever using the same testing dataset:
```
python3 -m colmap_retrieval.test data/test/colon/gnd_colon.pkl --load-score-path data/test/colon/vocab_tree_retrieval_score.bin
```
The code uses a software called COLMAP [(https://github.com/colmap/colmap)] to extract SIFT features and run vocabulary tree image retriever. Because this software is not easy to install, I precomputed the output of vocabulary tree retriever in data/test/colon/vocab-tree-retrieval-score.bin. The above program directly loads the scores instead of running COLMAP APIs. The .bin file is a numpy array of size (descriptor_size x num_queries). If this path is not specified, you need to install colmap to your system path, the script in this project calls the APIs directly.

cirtorch/examples/cnncolonclam.py is the implementation according to FABMAP work. However, because it does not produce satisfactory result, I am not attaching data for this file.

## Related publications
**Fine-tuning CNN Image Retrieval with No Human Annotation**,  
Radenović F., Tolias G., Chum O., 
TPAMI 2018 [[arXiv](https://arxiv.org/abs/1711.02512)]

**CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples**,  
Radenović F., Tolias G., Chum O., 
ECCV 2016 [[arXiv](http://arxiv.org/abs/1604.02426)]

**FAB-MAP: Probabilistic Localization and Mapping in the Space of Appearance**,
Mark Cummins and Paul Newman,
The International Journal of Robotics Research 2008 [[paper](http://www.robots.ox.ac.uk/~mjc/Papers/IJRR_2008_FabMap.pdf)]
