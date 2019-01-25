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
  --task supervised \
  --architecture resnet50 \
  --filters_factor 4 \
  --weight_decay 1e-4 \
  --dataset imagenet \
  --train_split trainval \
  --val_split test \
  --batch_size 256 \
  --eval_batch_size 10 \
  \
  --preprocessing inception_preprocess \
  --resize_size 224 \
  \
  --lr 0.1 \
  --lr_scale_batch_size 256 \
  --decay_epochs 30,60,80 \
  --epochs 90 \
  --warmup_epochs 5 \
  "$@"
