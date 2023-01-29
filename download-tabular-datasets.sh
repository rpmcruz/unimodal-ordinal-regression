#!/bin/bash
rm -rf data; mkdir data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data -P data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data -P data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data -P data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data -P data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data -P data
