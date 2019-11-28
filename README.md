# Programs written for my Beleg "Learning Depth Estimation from RGB and Infrared Input"
This repository contains all programs I've written for my "Großer Beleg" at the Technische Universität Dresden.
Languages utilized are `Python` (3.6) and `C++`. This documentation lists all important programs as well as their usage and requirements / dependencies. In general, the programs should be commented in a way that allows understanding them. Command line arguments can always be viewed by using `python Program.py --help`

## Learning/Model_VGG_Style.py
_**Description:**_ This python script contains the network architecture of the trained model. Input images are loaded with a custom data generator. Trains the model and creates a plot for the learning rate schedule. It also saves the training history as `pickle` file and the final model as `.h5` file.

_**Requirements:**_

| Package    | Link   |
|------------|------|
| Numpy      |  [link](https://anaconda.org/anaconda/numpy)    |
| Keras      |  [link](https://anaconda.org/conda-forge/keras)    |
| Matplotlib |  [link](https://anaconda.org/conda-forge/matplotlib)    |
| argparse   |  [link](https://anaconda.org/anaconda/argparse)    |
| OpenCV     |  [link](https://anaconda.org/conda-forge/opencv)    |

_**Command Line Arguments:**_

| Argument   | Description   |
|------------|---------------|
| -t, --train    | Path to folder that contains the training and validation examples   |
| -x, --output   | Path to folder where all output is saved to   |
| -b, --batch_size   | Batch size to train the network with   |
| -e, --epochs   | Number of epochs to train the network on   |
| -o, --optimizer   | The optimizer to utilize for training. Supported are SGD, Adam and RMSprop   |
| -l, --loss   | Loss function to utilize. Either MMAE, MMAE_simple, MRMSE or MRMSE_simple. Defaults to MMAE_simple  |
| -p, --periods   | Number of epochs after which to save the current model (and its weights). 1 means every epoch. Currently disabled due to bug that freezes the training process on Taurus.   |
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

## Visualization/DepthPredictor.py
_**Description:**_ Predicts a depth image for the provided images (see command line arguments). Outputs:
* Histograms of inputs and predictions (at the moment quite ugly)
* Colorization attempts for the predicted depth image (at the moment quite ugly)
* Difference Image, indicating if the prediction deviates more than a specified threshold from the ground truth (see command line arguments)
* Unprocessed* predicted depth image (* only clipped to [0,65535])

_**Requirements:**_

| Package    | Link   |
|------------|------|
| Numpy      |  [link](https://anaconda.org/anaconda/numpy)    |
| Keras      |  [link](https://anaconda.org/conda-forge/keras)    |
| Matplotlib |  [link](https://anaconda.org/conda-forge/matplotlib)    |
| argparse   |  [link](https://anaconda.org/anaconda/argparse)    |
| OpenCV     |  [link](https://anaconda.org/conda-forge/opencv)    |

_**Command Line Arguments:**_

| Argument   | Description   |
|------------|--------|
| -f, --folder   | If multiple depth images should be predicted, the color and infrared images, along with optional ground truth depth images, should be placed in a folder with subfolders 'Color', 'Infrared' and optionally 'Depth'. This is the path to this folder   |
| -c, --color   | If only a single depth image should be predicted, this is the corresponding color image   |
| -i, --infrared   | If only a single depth image should be predicted, this is the corresponding infrared image.   |
| -g, --ground_truth   | If only a single depth image should be predicted, this is the corresponding ground truth depth image. Can be ignored if no ground truth is available   |
| -b, --batch_size   | When multiple depth images should be predicted, this is the batch size that should be utilized while predicting   |
| -m, --model   | Path to the model that should be utilized for predicting depth images   |
| -o, --output   | Path to where the predicted images should be saved to   |
| --no_scaling   | Don't scale the input images to the range [0,1]   |
| --default_loss | Use the default mean absolute error loss funtion. Should not be utilized   |
| -t, --threshold_offset   | Offset for depth image normalization. Defaults to 2000. Only utilized if ground truth is given   |
| --old_model   | For old models, the loss function was called binary mean absolut error. Activate this if an 'Unknown loss function' error is thrown. Should not be utilized   |
| -d, --difference_threshold   | Utilized for difference visualization of ground truth and prediction. Maximum difference between ground truth and prediction in meters which is considered ok. Defaults to 0.05  |
| --depth_scale_text   | Text file containing depth scale of the utilized depth camera. Alternatively, use --depth_scale_value to directly provide a float   |
| --depth_scale_value   | Depth scale of the utilized depth camera. Alternatively, provide text file containing this scale with --depth_scale_text   |

## Visualization/PredictionPipeline.py
_**Description:**_ Used to predict depth images from a streaming RealSense depth camera or a prerecorded `rosbag` file. Additionally, a colorization of the predicted depth image is done (at the moment poorly). Note that the currently trained models are not fast enough to predict images in real time with a reasonable frame rate. The current implementation was not tested with a live streaming camera.

_**Requirements:**_ 

| Package    | Link   |
|------------|------|
| Numpy      |  [link](https://anaconda.org/anaconda/numpy)    |
| Keras      |  [link](https://anaconda.org/conda-forge/keras)    |
| pyrealsense2   |  [link](https://pypi.org/project/pyrealsense2/)    |
| argparse   |  [link](https://anaconda.org/anaconda/argparse)    |
| OpenCV     |  [link](https://anaconda.org/conda-forge/opencv)    |

_**Command Line Arguments:**_

| Argument   | Description   |
|------------|---------------|
| -p, --playback_path   | Path to a recorded sequence that should be predicted. Defaults to None (i.e. streaming configuration)   |
| --no_realtime   | Disables real time mode. Currently does nothing   |
| -m, --model   | Path to model that should be utilized for predictions   |
| --scale_output   | Scale the output of the network   |
| --no_clip   | Don't clip the output predictions to [0,65535]. Should not be utilized, since values outside of [0,65535] will under/overflow   |

## ImagePreprocessing/BadReading.py
_**Description:**_ This script demonstrates how to read frames from a recorded `rosbag` file in Python. To actually execute the script, uncomment the respective lines in the script.

_**Requirements:**_ 

| Package    | Link   |
|------------|------|
| Numpy      |  [link](https://anaconda.org/anaconda/numpy)    |
| pyrealsense2   |  [link](https://pypi.org/project/pyrealsense2/)    |
| argparse   |  [link](https://anaconda.org/anaconda/argparse)    |
| OpenCV     |  [link](https://anaconda.org/conda-forge/opencv)    |

_**Command Line Arguments:**_

| Argument   | Description   |
|------------|---------------|
| -i, --input   | Path the ROSBAG file that should be read in   |

## Visualization/Depth_Colorizer.ipynb
_**Description:**_ IPython Notebook that demonstrates how a 16-bit depth image can be colorized using OpenCV.

_**Requirements:**_ 

| Package    | Link   |
|------------|------|
| Numpy      |  [link](https://anaconda.org/anaconda/numpy)    |
| Matplotlib |  [link](https://anaconda.org/conda-forge/matplotlib)    | 
| OpenCV     |  [link](https://anaconda.org/conda-forge/opencv)    |

## ImagePreprocessing/ImagePreprocessing.py
_**Description:**_ This script is used to read frames from recorded `rosbag` files, apply preprocessing to them and save them in a folder structure that can be used for training the network. Execution of this script on a large file can take quite some time. There is also a `C++` script for this task, which is marginally faster.

_**Requirements:**_

| Package    | Link   |
|------------|------|
| Numpy      |  [link](https://anaconda.org/anaconda/numpy)    |
| pyrealsense2   |  [link](https://pypi.org/project/pyrealsense2/)    |
| argparse   |  [link](https://anaconda.org/anaconda/argparse)    |
| OpenCV     |  [link](https://anaconda.org/conda-forge/opencv)    |
| Tqdm   | [link](https://anaconda.org/conda-forge/tqdm)   |
| pathlib   | [link](https://anaconda.org/menpo/pathlib)   |

_**Command Line Arguments:**_

| Argument   | Description   |
|------------|---------------|
| -i, --input   | Path to main folder that contains subfolders 'Outdoor_Lighting' and 'Indoor_Lighting' which contain the recorded bag files   |
| -o, --output   | Path where the included images should be saved to   |
| -d, --decimation   | Decimation filter magnitude. Values of 2 and 3 perform median downsampling, values greater than 3 perform mean downsampling. Should be 1 if no decimation operation should be done   |
| -s, --skip   | Should scenes that already exist in the output location be skipped. Default is False. If activated, \_Part_2.bag files will not be considered. This can also be used to only process .bag files that are not yet processed   |
| -v, --verbose   | Verbosity settings. 0 - no extra messages. 1 - basic debug messages. 2 - additional runtime messages   |

## ImagePreprocessing/Image_Subsampling.py
_**Description:**_ Slightly modified version of Visualization/ImagePreprocessing.py. This script takes into account that only every x-th frame should be sampled from the `rosbag` files. It is also possible to define two files that should not be sampled from (for validation and test set).

_**Requirements:**_

| Package    | Link   |
|------------|------|
| Numpy      |  [link](https://anaconda.org/anaconda/numpy)    |
| pyrealsense2   |  [link](https://pypi.org/project/pyrealsense2/)    |
| argparse   |  [link](https://anaconda.org/anaconda/argparse)    |
| OpenCV     |  [link](https://anaconda.org/conda-forge/opencv)    |
| Tqdm   | [link](https://anaconda.org/conda-forge/tqdm)   |
| pathlib   | [link](https://anaconda.org/menpo/pathlib)   |

_**Command Line Arguments:**_

| Argument   | Description   |
|------------|---------------|
| -i, --input   | Path to main folder that contains subfolders 'Outdoor_Lighting' and 'Indoor_Lighting' which contain the recorded bag files   |
| -o, --output   | Path where the included images should be saved to   |
| -s, --subsample   | The subsampling rate   |
| -v, --verbose   | Verbosity settings. 0 - no extra messages. 1 - basic debug messages. 2 - additional runtime messages   |
| -1, --skip_test   | The first scene that should not be sampled from. This scene will be used as test data   |
| -2, --skip_validation   | The second scene that should not be sampled from. This scene will be used as validation data   |

