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

# About the net

The train downloads from ImageNet database images of the following classes/synset (defined in `lib/dataset/imagenet.py`) <br/>
```python
CLASSES = {'synthesizer':'n04376400', 'pipe organ':'n03854065', 'music box': 'n03801353', 'electric guitar':'n03272010', 'sax':'n04141076', 'ocarina':'n03840681', 'harmonica':'n03494278', 'acoustic guitar':'n02676566', 'trombone':'n04487394','gong':'n03447721', 'maraca':'n03720891', 'xylophone':'n03721384', 'pianoforte':'n03928116'}
```

If you want to change the classes you should change it and the tuple after it. Also you must put the extracted annotations in `data/imagenet/Annotation_imagenet` (pull requests are welcome).

## Results

After a session of 10000 iterations (it took less than 1 day on a Nvidia GTX 980) these are the results (obtained by running `demo.py`)

![Alt text](samples/sample1.png?raw=true "Sample")
![Alt text](samples/sample2.png?raw=true "Sample")
![Alt text](samples/sample3.png?raw=true "Sample")
![Alt text](samples/sample4.png?raw=true "Sample")
![Alt text](samples/sample5.png?raw=true "Sample")
![Alt text](samples/sample6.png?raw=true "Sample")

while these are the metrics visualized on tensorboard

![Alt text](samples/loss.png?raw=true "Sample")
![Alt text](samples/cross_entropy.png?raw=true "Sample")
