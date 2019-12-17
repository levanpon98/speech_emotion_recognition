

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
