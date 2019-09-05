#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, Linear Labs Technologies

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from setuptools import find_packages, setup,command
with open("README.md", "r") as fp:
    long_description = fp.read()

with open('requirements.txt','r') as fp:
    requirements = [l.strip() for l in fp.readlines()]
# class install(command.install.install):
#     def run(self):
#         command.install.install.run(self)

p = find_packages()
setup(
    name="BLOX",
    packages=p,
    scripts=['blox' ],
    description="Neural Network building blocks. A simple and extensible wrapper around pytorch",
    long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://linearlabs.io",
    author="Ted Troxell",
    # cmdclass={'install':install},
    zip_safe=False,
    install_requires=requirements, # some cannot be installed via easy_install and will need to do it in install script
)