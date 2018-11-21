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
## Related publications

### Training (fine-tuning) convolutional neural networks 
```
@article{RTC18,
 title = {Fine-tuning {CNN} Image Retrieval with No Human Annotation},
 author = {Radenovi{\'c}, F. and Tolias, G. and Chum, O.}
 journal = {TPAMI},
 year = {2018}
}
```
```
@inproceedings{RTC16,
 title = {{CNN} Image Retrieval Learns from {BoW}: Unsupervised Fine-Tuning with Hard Examples},
 author = {Radenovi{\'c}, F. and Tolias, G. and Chum, O.},
 booktitle = {ECCV},
 year = {2016}
}
```
