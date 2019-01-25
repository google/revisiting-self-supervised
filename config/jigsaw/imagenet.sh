#!/bin/bash -eu
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/sh

python train_and_eval.py \
  --task jigsaw \
  --dataset imagenet \
  --train_split train \
  --val_split val \
  --batch_size 128 \
  --eval_batch_size 8 \
  \
  --architecture resnet50 \
  --filters_factor 8 \
  --last_relu True \
  --mode v1 \
  \
  --preprocessing resize,to_gray,crop,crop_patches,standardization \
  --resize_size 292,292 \
  --crop_size 255 \
  --grayscale_probability 0.66 \
  --splits_per_side 3 \
  --patch_jitter 21 \
  --embed_dim 1000 \
  \
  --lr 0.1 \
  --lr_scale_batch_size 256 \
  --decay_epochs 15,25 \
  --epochs 35 \
  --warmup_epochs 5 \
  \
  --serving_input_shape None,64,64,3 \
  "$@"
