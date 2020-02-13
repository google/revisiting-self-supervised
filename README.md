# Revisiting self-supervised visual representation learning

Tensorflow implementation of experiments from
[our paper on unsupervised visual representation learning](http://arxiv.org/abs/1901.09005).

If you find this repository useful in your research, please consider citing:

```
@inproceedings{kolesnikov2019revisiting,
    title={Revisiting self-supervised visual representation learning},
    author={Kolesnikov, Alexander and Zhai, Xiaohua and Beyer, Lucas},
    journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2019}
}
```

## Overview

This codebase allows to reproduce core experiments from our paper. It contains
our re-implementation of four self-supervised representation learning
techniques, utility code for running training and evaluation loops (including on
TPUs) and an implementation of standard CNN models, such as ResNet v1, ResNet v2
and VGG19.

Specifically, we provide a re-implementation of the following self-supervised
representation learning techniques:

1.  [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)
2.  [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)
3.  [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)
4.  [Discriminative Unsupervised Feature Learning with Exemplar Convolutional
    Neural Networks](https://arxiv.org/abs/1406.6909)

## Usage instructions

In the paper we train self-supervised models using 32 or 128 TPU cores. We
evaluate the resulting representations by training a logistic regression model
on 32 TPU cores.

In this codebase we provide configurations for training/evaluation of our models
using an 8 TPU core setup as this setup is more affordable for public TPU users
through the Google Cloud API. These configurations produce results close to those
reported in the paper, which used more TPU chips.

For debugging or running small experiments we also support training and
evaluation using a single GPU device.

### Preparing data

Please refer to the
[instructions in the slim library](https://github.com/tensorflow/models/blob/master/research/inception/README.md#getting-started)
for downloading and preprocessing ImageNet data.

### Clone the repository and install dependencies

```
git clone https://github.com/google/revisiting-self-supervised
cd revisiting-self-supervised
python -m pip install -e . --user
```

We depend on some external files that need to be downloaded and placed in the
root repository folder. You can run the following commands to download them:

```
wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/preprocessing/inception_preprocessing.py
wget https://github.com/MehdiNoroozi/JigsawPuzzleSolver/raw/master/permutations_100_max.bin
```

### Running locally on a single GPU

Run any experiment by running the corresponding shell script with the following
options, here exemplified for the fully supervised experiment:

```
./config/supervised/imagenet.sh \
  --workdir <WORKING_DIRECTORY> \
  --nouse_tpu \
  --master='' \
  --dataset_dir <PREPROCESSED_IMAGENET_PATH>
```

### Running on Google Cloud using TPUs

#### Step 1:

Create your own TPU cloud instance by following the
[official documentation](https://cloud.google.com/tpu/).

#### Step 2:

Clone the repository and install dependencies as described above.

#### Step 3:

Run the self supervised model training script with TPUs. For example:

```
gsutil mb gs://<WORKING_DIRECTORY>
export TPU_NAME=<TPU_PROJECT_NAME>
config/supervised/imagenet.sh --workdir gs://<WORKING_DIRECTORY> --dataset_dir gs://<PREPROCESSED_IMAGENET_PATH>
```

After/during training, run the self supervised model evaluation script with
TPUs. It generates the loss and metric on the validation set, and exports a hub
module under directory `gs://<WORKING_DIRECTORY>/export/hub/<TIMESTAMP>/module`:

```
config/supervised/imagenet.sh --workdir gs://<WORKING_DIRECTORY> --dataset_dir gs://<PREPROCESSED_IMAGENET_PATH> --run_eval
```

Note, that `<TPU_PROJECT_NAME>` is set by the user when creating the Cloud TPU
node. Moreover, ImageNet data and the working directory should be placed in a
Google Cloud bucket storage.

#### Step 4:

Evaluates the self supervised models with logistic regression. You need to pass
the exported hub module from step 3 above as an additional argument:

```
gsutil mb gs://<EVAL_DIRECTORY>
export TPU_NAME=<TPU_PROJECT_NAME>
config/evaluation/rotation_or_exemplar.sh --workdir gs://<EVAL_DIRECTORY> --dataset_dir gs://<PREPROCESSED_IMAGENET_PATH> --hub_module gs://<PATH_TO_YOUR_HUB_MODULE>

config/evaluation/rotation_or_exemplar.sh --workdir gs://<EVAL_DIRECTORY> --dataset_dir gs://<PREPROCESSED_IMAGENET_PATH> --hub_module gs://<PATH_TO_YOUR_HUB_MODULE> --run_eval
```

You could start a tensorboard to visualize the training/evaluation progress:

```
tensorboard --port 2222 --logdir gs://<EVAL_DIRECTORY>
```

## Pretrained models

If you want to download and try our best self-supervised models please see this [Ipython
notebook](https://colab.research.google.com/drive/1HdApkScZpulQrACrPKZiKYHhy7MeR3iN).


## Authors

- [Alexander Kolesnikov](https://github.com/kolesman)
- [Xiaohua Zhai](https://sites.google.com/site/xzhai89/)
- [Lucas Beyer](http://lucasb.eyer.be/)
- [Marvin Ritter](https://github.com/Marvin182)

### This is not an official Google product
