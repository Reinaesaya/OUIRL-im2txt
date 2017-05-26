# OUIRL (Osaka University Intelligent Robotics Laboratory) im2txt

Experimentation upon im2txt neural network model to process images/videos and hopefully generate conversation, in concurrence with a [chatterbot](https://github.com/Reinaesaya/OUIRL-ChatBot). Generating captions with video has rather tremendous lag and is pretty slow.

- Contains a copy of the [im2txt repository](https://github.com/tensorflow/models/tree/master/im2txt). Please refer to this for detailed descriptions and instructions
- Extra shell scripts made in reference to [im2txt_demo](https://github.com/siavash9000/im2txt_demo)
- Pretrained models from [KranthiGV](https://github.com/KranthiGV/Pretrained-show-and-Tell-model)

## Setup Overview

### Install Required Packages
First ensure that you have installed the following required packages (its suggest that you use anaconda2 to conda install tensorflow, numpy, and nltk):

* **Bazel** ([instructions](http://bazel.io/docs/install.html))
* **TensorFlow** 1.0 or greater ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](http://www.scipy.org/install.html))
* **Natural Language Toolkit (NLTK)**:
    * First install NLTK ([instructions](http://www.nltk.org/install.html))
    * Then install the NLTK data ([instructions](http://www.nltk.org/data.html))
* **OpenCV**
    * Must have OpenCV version with ffmpeg enabled, or else ```cv2.VideoCapture``` command will not work

### Download Pretrained Data

1) Download the [2M iterations finetuned checkpoint file](https://drive.google.com/file/d/0B3laN3vvvSD2T1RPeDA5djJ6bFE/view?usp=sharing) into pretrained_model folder | [Released under MIT License](https://github.com/KranthiGV/Pretrained-Show-and-Tell-model/blob/master/LICENSE)

2) Change the checkpoint\_path and vocab\_file in [process_image.sh](process_image.sh) and [process_vid.sh](process_vid.sh) to point to the location of your pretrained model

### Build Inference

```
./build_inference.sh
```

### Running

For generating picture for a single caption

```
./process_image.sh /path/to/image.jpg
```

