# SPEXAI: A Neural Network Emulator for SPEX

This package contains code to both train and use neural networks as an emulator 
(or surrogate model) for the Collisional Ionization Equilibrium (CIE) model in SPEX.
At the moment, only this model is supported, though we anticipate building emulators 
for more complex models in the future. 

There are two separate parts to the package: the `train` submodule contains all the 
code to (re-)train the emulator for each element.

The emulator was trained in PyTorch, and for using it directly to fit X-ray spectra, we 
provide a pre-trained model that can be used as a drop-in replacement for the CIE model 
in SPEX. In the `inference` submodule, we provide functionality to use the model to 
fit data directly. 

**Note**: At the moment, this includes effects like redshift, turbulent velocity 
broadening and instrument responses, but *not* other X-ray model components such as 
interstellar absorption. More to come! 


## Required Data

Some of the files necessary for training or using the emulator are too big for a GitHub repository.
Depending of how you're planning to use the code. 

* Training Data: 
* Supporting files for inference:  

## Installation Instructions 

To install this package, follow the instructions below.

### Dependencies

The package requires the following dependencies:
* numpy
* pandas
* torch
* scikit-learn
* matplotlib

For the inference submodule, additional requirements are:
* scipy
* emcee
* seaborn
* astropy
* torchinterp1d: https://github.com/aliutkus/torchinterp1d

All dependencies except the `torchinterp1d` package should be installable via pip or 
conda. We suggest working in a clean virtual environment. 

To install `torchinterp1d`, please clone or download the repository, then install it with
`pip install -e` in the root folder of that directory in the same environment as the 
packages above. 

### Installing SpexAI

## Authors
* Jip Matthijsse
* Daniela Huppenkothen

 
