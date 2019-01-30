#!/bin/bash

cd data/coco/PythonAPI 
python setup.py build_ext --inplace 
python setup.py build_ext install

cd ../../../lib/utils
python setup.py build_ext --inplace 
