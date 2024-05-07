#!/bin/bash

DATASETS="ICIAR HCI FGNET SMEAR2005 FOCUSPATH"
LOSSES="CrossEntropy POM OrdinalEncoding BinomialUnimodal_CE PoissonUnimodal CO2 HO2 CDW_CE UnimodalNet WassersteinUnimodal_Wass WassersteinUnimodal_KLDIV"
FOLDS=`seq 0 4`

echo "Dataset & Loss & Acc (\%) & QWK (\%) & MAE & \%Unimodal \\\\"
for DATASET in $DATASETS; do
for LOSS in $LOSSES; do
python3 test.py $DATASET $LOSS --reps $FOLDS --data /data/ordinal
echo "\\\\"
done
done
