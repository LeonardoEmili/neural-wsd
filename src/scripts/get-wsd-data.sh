#!/bin/bash

DATASET_DIRECTORY="data/"
envfile="../.env"
mkdir -p $DATASET_DIRECTORY

# Retrieve the train data for WSD (Raganato et al., 2017)
WSD_TRAIN_DATA_NAME="WSD_Training_Corpora"
WSD_TRAIN_URL="https://github.com/andreabac3/neural-wsd/releases/download/1.0/WSD_Training_Corpora.zip"
WSD_TRAIN_DATA_ZIP=$DATASET_DIRECTORY+$WSD_TRAIN_DATA_NAME".zip"
WSD_TRAIN_DATA=$DATASET_DIRECTORY+$WSD_TRAIN_DATA_NAME

if [ -f "$envfile" ]; then
    # Using GitHub API to fetch the WSD datasets (usually faster)
    bash src/scripts/github_downloader.sh $WSD_TRAIN_DATA_NAME".zip" $WSD_TRAIN_DATA_ZIP
else
    # Using plain download from origin servers whenever credentials are missing
    WSD_TRAIN_URL="http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip"
    curl $WSD_TRAIN_URL -o $WSD_TRAIN_DATA_ZIP
fi

unzip $WSD_TRAIN_DATA_ZIP -d $DATASET_DIRECTORY
rm $WSD_TRAIN_DATA_ZIP

# Retrieve the evaluation data for WSD (Raganato et al., 2017)
WSD_EVAL_DATA_NAME="WSD_Unified_Evaluation_Datasets"
WSD_EVAL_URL="https://github.com/andreabac3/neural-wsd/releases/download/1.0/WSD_Unified_Evaluation_Datasets.zip"
WSD_EVAL_DATA_ZIP=$DATASET_DIRECTORY+$WSD_EVAL_DATA_NAME".zip"
WSD_EVAL_DATA=$DATASET_DIRECTORY+$WSD_EVAL_DATA_NAME

if [ -f "$envfile" ]; then
    # Using GitHub API to fetch the WSD datasets (usually faster)
    bash src/scripts/github_downloader.sh $WSD_EVAL_DATA_NAME".zip" $WSD_EVAL_DATA_ZIP
else
    # Using plain download from origin servers whenever credentials are missing
    WSD_EVAL_URL="http://lcl.uniroma1.it/wsdeval/data/WSD_Unified_Evaluation_Datasets.zip"
    curl $WSD_EVAL_URL -o $WSD_EVAL_DATA_ZIP
fi

unzip $WSD_EVAL_DATA_ZIP -d $DATASET_DIRECTORY
rm $WSD_EVAL_DATA_ZIP