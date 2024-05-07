#!/bin/bash

DATASETS="ICIAR HCI FGNET SMEAR2005 FOCUSPATH"
#DATASETS="ABALONE5 ABALONE10 BALANCE_SCALE CAR NEW_THYROID"
LOSSES="WassersteinUnimodal_Wass"
LAMDAS="0.001 0.01 0.1 1 10 100 1000"

echo "\documentclass{standalone}"
echo "\usepackage{xcolor}"
echo "\begin{document}"
echo "\begin{tabular}{lllll}"
echo "\bf Method & \bf Lambda & \bf acc & \bf mae & \bf unimodal \\\\\\hline"

for DATASET in $DATASETS; do
    echo "\\bf $DATASET \\\\"
    for LOSS in $LOSSES; do
        for LAMDA in $LAMDAS; do
            python3 test.py $DATASET $LOSS --lamda $LAMDA --reps 0 --print-lambda --datadir /data/ordinal
        done
    done
    echo "\hline"
done

echo "\end{tabular}"
echo "\end{document}"
