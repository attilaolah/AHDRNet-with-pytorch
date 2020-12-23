#!/usr/bin/env bash

mkdir -p data
pushd data
for dataset in Training Test; do
  wget --continue "https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR_${dataset}set.zip"
  unzip -n "SIGGRAPH17_HDR_${dataset}set.zip"
  rm "${dataset}/README.txt"
done
popd
