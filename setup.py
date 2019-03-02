import setuptools

import os
import sys
import zipfile

# unpack splits to dir in home
home_dir = os.path.expanduser("~")
splits_dir = os.path.join(home_dir, ".dstorch_splits")
if not os.path.isdir(splits_dir):
    os.makedirs(splits_dir)

zip_ref = zipfile.ZipFile(os.path.join('dstorch', '__data__', 'splits.zip'), 'r')
zip_ref.extractall(splits_dir)
zip_ref.close()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dstorch-yassnda",
    version="0.0.1",
    author="Durasov Nikita",
    author_email="yassnda@gmail.com",
    description="Package for convenient usage of popular computer vision datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NikitaDurasov/dstorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)