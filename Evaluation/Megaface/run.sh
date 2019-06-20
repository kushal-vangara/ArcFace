#!/usr/bin/env bash

DEVKIT="/afs/crc.nd.edu/user/v/valbiero/MegaFace/devkit/experiments"
ALGO="r50-combined-vggface2"
ROOT=$(dirname `which $0`)
echo $ROOT
python2 -u gen_megaface.py --gpu 0 --algo "$ALGO" --model '/afs/crc.nd.edu/user/v/valbiero/ext_vol2/training/mxnet/r50-combined-vggface2/model,1'
python2 -u remove_noises.py --algo "$ALGO"

cd "$DEVKIT"
python2 -u run_experiment.py "$ROOT/feature_out_clean/megaface" "$ROOT/feature_out_clean/facescrub" _"$ALGO".bin ../../mx_results/ -s 1000000 -p ../templatelists/facescrub_features_list.json
