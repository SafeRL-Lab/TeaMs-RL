#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
import setuptools

from codecs import open
from os import path

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("teams_rl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

import os
from setuptools import setup, find_packages
import setuptools



here = path.abspath(path.dirname(__file__))


extras = {}

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

long_description = 'TODO.'

setup(
    name='pcrpo',
    version=get_version(),
    description='Teams-RL',
    long_description=long_description,
    author="authors",
    author_email='gmail',
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith('node')],
    zip_safe=False,
    # install_requires=requires_list,
    extras_require=extras,
    python_requires=">=3.8",
    # PyPI package information.
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 ]

)
