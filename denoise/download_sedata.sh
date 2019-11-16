
mkdir -p dataset_zip
pushd dataset_zip

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
popd

mkdir -p dataset_tmp
pushd dataset_tmp
unzip -q -j ../dataset_zip/clean_trainset_28spk_wav.zip -d trainset_clean
unzip -q -j ../dataset_zip/noisy_trainset_28spk_wav.zip -d trainset_noisy
unzip -q -j ../dataset_zip/clean_testset_wav.zip -d valset_clean
unzip -q -j ../dataset_zip/noisy_testset_wav.zip -d valset_noisy
popd

declare -a arr=("trainset_clean" "trainset_noisy" "valset_clean" "valset_noisy")

mkdir -p dataset
pushd dataset_tmp
for d in */; do
    mkdir -p "../dataset/$d"
    pushd "$d"
    for f in *.wav; do
        sox "$f" -e float -b 32 "../../dataset/$d$f" rate -v -I 16000
    done
    popd
done
popd

rm -r dataset_tmp
