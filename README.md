# tf-faster-rcnn
Tensorflow Faster R-CNN <b>for Windows</b> by using Python 3.5. By default it uses Imagenet database.

This is the repository to compile Faster R-CNN on Windows. It is heavily inspired by the great work done [here](https://github.com/smallcorgi/Faster-RCNN_TF) and [here](https://github.com/rbgirshick/py-faster-rcnn). I have not implemented anything new but I fixed the implementations for Windows and Python 3.5.

# Installation
1- Install tensorflow, preferably GPU version. Follow [instructions]( https://www.tensorflow.org/install/install_windows).

2- Install python packages (cython, python-opencv, easydict)

3- Clone this repository

4- Move to `data/coco/PythonAPI` and launch
```
python setup.py build_ext --inplace 
python setup.py build_ext install
```
Then in `lib/utils`
```
python setup.py build_ext --inplace 
```

5- Download pre-trained VGG16 from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) and place it as "data\imagenet_weights\vgg16.ckpt"
 
 For rest of the models, please check [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
 
6- Run train.py
  
Notify me if there is any issue
 
