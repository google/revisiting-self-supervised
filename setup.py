#!/usr/bin/python
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

"""Install self supervised learning package."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='self_supervised_learning',
    version='1.0',
    description=('Self Supervised Learning - code from "Revisiting '
                 'Self-Supervised Visual Representation Learning" paper'),
    author='Google LLC',
    author_email='no-reply@google.com',
    url='http://github.com/TODO',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={
    },
    scripts=[
    ],
    install_requires=[
        'future',
        'numpy',
        'absl-py',
        'tensorflow',
        'tensorflow-hub',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow self supervised learning',
)
