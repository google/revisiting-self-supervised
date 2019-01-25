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

# For the reported results, try following parameters on 4x4 TPUs:
#   batch_size: 2048
#   decay_epochs: 480,500
#   epochs: 520
python train_and_eval.py \
  --task linear_eval \
  --dataset imagenet \
  --train_split trainval \
  --val_split test \
  --batch_size 512 \
  --eval_batch_size 32 \
  \
  --preprocessing resize_small,crop,crop_patches,standardization \
  --resize_size 256,256 \
  --crop_size 192,192 \
  --smaller_size 224 \
  --patch_jitter 0 \
  --splits_per_side 3 \
  --pool_mode max \
  --combine_patches avg_pool \
  \
  --lr 0.1 \
  --decay_epochs 30,50 \
  --epochs 70 \
  --lr_scale_batch_size 256 \
  --hub_module ~/Downloads/module \
  "$@"
