#!/bin/bash

## DOWNLOAD THE DATASETS
mkdir -p dataset_zip
cd dataset_zip
# TRAINING DATASET
if [ ! -f clean_trainset_28spk_wav.zip ]; then
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip
fi
if [ ! -f noisy_trainset_28spk_wav.zip ]; then
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip
fi
# VALIDATION DATASET
if [ ! -f clean_testset_wav.zip ]; then
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip
fi
if [ ! -f noisy_testset_wav.zip ]; then
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip
fi
cd ..

## INFLATE DATA
mkdir -p dataset_tmp
cd dataset_tmp
unzip -q -j ../dataset_zip/clean_trainset_28spk_wav.zip -d trainset_clean
unzip -q -j ../dataset_zip/noisy_trainset_28spk_wav.zip -d trainset_noisy
unzip -q -j ../dataset_zip/clean_testset_wav.zip -d valset_clean
unzip -q -j ../dataset_zip/noisy_testset_wav.zip -d valset_noisy
cd ..

## RESAMPLE
declare -a arr=("trainset_clean" "trainset_noisy" "valset_clean" "valset_noisy")

mkdir -p dataset
cd dataset_tmp
for d in */; do
    mkdir -p "../dataset/$d"
    cd "$d"
    for f in *.wav; do
        sox "$f" -e float -b 32 "../../dataset/$d$f" rate -v -I 16000
    done
    cd ..
done
cd ..

# REMOVE TMP DATA
rm -r dataset_tmp