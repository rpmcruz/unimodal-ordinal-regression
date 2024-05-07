#!/bin/bash

DATASETS="ICIAR HCI FGNET SMEAR2005 FOCUSPATH"
LOSSES="CrossEntropy POM OrdinalEncoding BinomialUnimodal_CE PoissonUnimodal CO2 HO2 CDW_CE UnimodalNet WassersteinUnimodal_Wass WassersteinUnimodal_KLDIV"
FOLDS=`seq 0 4`

for FOLD in $FOLDS; do
for DATASET in $DATASETS; do
for LOSS in $LOSSES; do
sbatch python3 train.py $DATASET $LOSS $FOLD --datadir /data/ordinal
done
done
done
