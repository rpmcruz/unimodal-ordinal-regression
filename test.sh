#!/bin/bash

TABULAR_DATASETS="ABALONE5 ABALONE10 BALANCE_SCALE CAR NEW_THYROID"
IMAGE_DATASETS="HCI ICIAR FGNET SMEAR2005 FOCUSPATH"
LOSSES="CrossEntropy POM OrdinalEncoding CrossEntropy_UR CDW_CE BinomialUnimodal_CE PoissonUnimodal UnimodalNet"
LOSSES_LAMBDA="CO2 HO2 WassersteinUnimodal_KLDIV WassersteinUnimodal_Wass"

for DATASETS_TYPE in "TABULAR" "IMAGE"; do
if [ "$DATASETS_TYPE" == "TABULAR" ]; then
DATASETS=$TABULAR_DATASETS
DATADIR=/data/ordinal/UCI
else
DATASETS=$IMAGE_DATASETS
DATADIR=/data/ordinal
fi
echo $DATASETS_TYPE
echo "\documentclass{standalone}"
echo "\usepackage{xcolor}"
echo "\begin{document}"
echo "\begin{tabular}{llllllllllllll}"
for DATASET in $DATASETS; do
    echo -n "\\bf $DATASET" # & \bf Accuracy & \bf MAE & \bf Times Unimodal \\\\"
    for LOSS in $LOSSES; do echo -n " & $LOSS"; done
    for LOSS in $LOSSES_LAMBDA; do echo -n " & $LOSS"; done
    echo " \\\\"

    for METRIC in $(seq 0 6); do
        if [ $METRIC -eq 0 ]; then echo -n "\%Accuracy"; fi
        if [ $METRIC -eq 1 ]; then echo -n "QWK"; fi
        if [ $METRIC -eq 2 ]; then echo -n "MAE"; fi
        if [ $METRIC -eq 3 ]; then echo -n "\%Unimodal"; fi
        if [ $METRIC -eq 4 ]; then echo -n "ZME"; fi
        if [ $METRIC -eq 5 ]; then echo -n "NLL"; fi
        if [ $METRIC -eq 6 ]; then echo -n "\%$\tau$"; fi
        for LOSS in $LOSSES; do
            python3 test.py $DATASET $LOSS --reps 1 2 3 4 --only-metric $METRIC --datadir $DATADIR
        done
        for LOSS in $LOSSES_LAMBDA; do
            LAMBDA=`python3 test-best-lambda.py $DATASET $LOSS --datadir $DATADIR`
            python3 test.py $DATASET $LOSS --reps 1 2 3 4 --lamda $LAMBDA --only-metric $METRIC --datadir $DATADIR
        done
        echo " \\\\"
    done
    echo "\\hline"
done
echo "\\end{tabular}"
echo "\\end{document}"
done