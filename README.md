# Programs written for my Beleg "Learning Depth Estimation from RGB and Infrared Input"
This repository contains all programs I've written for my "Großer Beleg" at the Technische Universität Dresden.
Languages utilized are `Python` (3.6) and `C++`. This documentation lists all important programs as well as their usage and requirements / dependencies.

## Learning/Model_VGG_Style.py
_**Description:**_ This python script contains the network architecture of the trained model. To train 
_**Requirements:**_

| Package    | Link   |
|------------|------|
| Numpy      |  [link](https://anaconda.org/anaconda/numpy)    |
| Keras      |  [link](https://anaconda.org/conda-forge/keras)    |
| Matplotlib |  [link](https://anaconda.org/conda-forge/matplotlib)    |
| argparse   |  [link](https://anaconda.org/anaconda/argparse)    |
| OpenCV     |  [link](https://anaconda.org/conda-forge/opencv)    |

_**Command Line Arguments:**_ It is also possible to use the --help parameter to get a list containing all command line arguments

| Argument   | Description   |
|------------|---------------|
| -t, --train    | Path to folder that contains the training and validation examples   |
| -x, --output   | Path to folder where all output is saved to   |
| -b, --batch_size   | Batch size to train the network with   |
| -e, --epochs   | Number of epochs to train the network on   |
| -o, --optimizer   | The optimizer to utilize for training. Supported are SGD, Adam and RMSprop   |
| -l, --loss   | Loss function to utilize. Either MMAE, MMAE_simple, MRMSE or MRMSE_simple. Defaults to MMAE_simple  |
| -p, --periods   | Number of epochs after which to save the current model (and its weights). 1 means every epoch   |
| -d, --decay   | Reduce learning rate after every x epochs. Defaults to 10   |
| -f, --factor_decay   | Factor to reduce the learning rate. Defaults to 0.5   |
| --default_optimizers   | Enable all keras optimizers, not only SGD, Adam and RMSprop. This will deactivate learning rate decay   |
| --omit_batchnorm   | Don't add batch normalization layers after convolutions   |
| -m, --momentum   | Momentum used in batch normalization layers. Defaults to 0.99. If validation loss oscillates, try lowering it (e.g. to 0.6)   |
| --skip_0   | Functionality of S0 skip connections. One of the following: 'add', 'concat', 'concat+' or 'disable'. Defaults to 'add'. 'Concat+' adds convolutions after concatenating   |
| --sgd_momentum   | Only works when using SGD optimizer: Not specified/'None': no momentum, 'normal': momentum with value from --sgd_momentum_value, 'nesterov': Use nesterov momentum with value from --sgd_momentum_value   |
| --sgd_momentum_value   | Only works when using SGD optimizer: Momentum value for SGD optimizer. Enable by using --sgd_momentum. Defaults to 0.9   |
| --output_sigmoid_activation   | Adds an sigmoid activation function to the output layer. The provided argument defines whether the ground truth is scaled to also fit this interval ('scale_input') or if the predictions get scaled in the loss function ('scale_output'). Defaults to '' (no sigmoid activation function added)   |
| --no_shuffle   | Disables shuffling of batches for each epoch   |
| --no_scale   | Disables scaling of input images to the range of [0,1]   |
