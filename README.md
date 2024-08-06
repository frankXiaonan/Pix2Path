# Pix2Path

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pix2Path (Pixel-level spatial transcriptomics to digital pathology) is a method that utilizes high-resolution spatial transcriptomic data to map potential disease progression and predict the effects of individual genes.

<p align="center">
  <img src="/docs/Pix2Path.png" width="100%"/>
</p>
## System requirements
### Operating system
The software has been tested on the CentOS Linux 7 system.

### Software requirements
- python 3.11</br>
- numpy </br>
- pandas </br>
- spatialdata </br>
- spatialdata-io </br>
- spatialdata-plot </br>
- ipython </br>
- matplotlib </br>
- tensorflow </br>
- scipy </br>

### Installation
pip install -r requirements.txt

### Usage - how to run predictor
cd src
python3 ad_pathology_predictor.py

### Output
/images: images of plt.show()
/data: datasets
/src: source files
/logs: checkpoints of model training

### Manually Load Checkpoints an display in Tensor Board
tensorboard --logdir logs/fit

### Contact
Contact us if you have any questions:</br>
Xiaonan Fu and Yan Chen: xnfu at uw.edu</br>


### License
This project is licensed under the MIT License - see the LICENSE file for details.
