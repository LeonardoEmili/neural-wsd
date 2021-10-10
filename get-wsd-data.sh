#!/bin/bash

DATASET_DIRECTORY="data/"
mkdir -p $DATASET_DIRECTORY

# Retrieve the train data for WSD (Raganato et al., 2017)
WSD_TRAIN_URL="http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip"
WSD_TRAIN_DATA_ZIP=$DATASET_DIRECTORY+"WSD_Training_Corpora.zip"
WSD_TRAIN_DATA=$DATASET_DIRECTORY+"WSD_Training_Corpora"

curl $WSD_TRAIN_URL -o $WSD_TRAIN_DATA_ZIP
unzip $WSD_TRAIN_DATA_ZIP -d $DATASET_DIRECTORY
rm $WSD_TRAIN_DATA_ZIP
