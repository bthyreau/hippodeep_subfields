# hippodeep-subfields
Brain Hippocampus Segmentation with Subfields

This program can quickly segment (<1min) the Hippocampus of raw brain T1 images.

This program has not been validated. It is not recommended to use it.

## Requirement

This program requires Python 3, with the PyTorch library, version > 1.0.0.

No GPU is required

ANTs is also necessary, as the program currently calls 'antsApplyTransforms'

Tested on Linux CentOS 6.x/7.x, Ubuntu 18.04 and MacOS X 10.13, using PyTorch versions 1.0.0 to 1.4.0
In addition, Windows compatibility patches by Bernd Foerster are available at https://github.com/bfoe/hippodeep_subfields

## Installation

Just clone or download this repository.

In addition to PyTorch, the code requires scipy and nibabel.

The simplest way to install from scratch is maybe to use a Anaconda environment, then
* install scipy (`conda install scipy` or `pip install scipy`) and  nibabel (`pip install nibabel`)
* get pytorch for python from `https://pytorch.org/get-started/locally/`. CUDA is not necessary.


## Usage:
To use the program, simply call:

`deepseg.sh example_brain_t1.nii.gz`.

Use -h for usage, in particular, -d to keep higher-resolutions images
