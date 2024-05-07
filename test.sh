#!/bin/bash

DATASETS="ICIAR HCI FGNET SMEAR2005 FOCUSPATH"
LOSSES="DummyModel CrossEntropy POM OrdinalEncoding CrossEntropy_UR CDW_CE BinomialUnimodal_CE PoissonUnimodal UnimodalNet"
LOSSES_LAMBDA="WassersteinUnimodal_KLDIV WassersteinUnimodal_Wass CO2 HO2"

echo "\documentclass{standalone}"
echo "\usepackage{xcolor}"
echo "\begin{document}"
echo "\begin{tabular}{llllllllllll}"
#echo "\bf Method & \bf Accuracy & \bf MAE & \bf Times Unimodal \\\\\\hline"

for DATASET in $DATASETS; do
    echo -n "\\bf $DATASET" # & \bf Accuracy & \bf MAE & \bf Times Unimodal \\\\"
    for LOSS in $LOSSES; do echo -n " & $LOSS"; done
    for LOSS in $LOSSES_LAMBDA; do echo -n " & $LOSS"; done
    echo " \\\\"

    for METRIC in $(seq 0 2); do
        if [ $METRIC -eq 0 ]; then echo -n "\%Accuracy"; fi
        if [ $METRIC -eq 1 ]; then echo -n "MAE"; fi
        if [ $METRIC -eq 2 ]; then echo -n "\%Unimodal"; fi
        for LOSS in $LOSSES; do
            python3 test.py $DATASET $LOSS --reps 1 2 3 4 --datadir /data/ordinal --only-metric $METRIC
        done
        for LOSS in $LOSSES_LAMBDA; do
            LAMBDA=`python3 test-best-lambda.py $DATASET $LOSS --datadir /data/ordinal`
            python3 test.py $DATASET $LOSS --reps 1 2 3 4 --lamda $LAMBDA --datadir /data/ordinal --only-metric $METRIC
        done
        echo " \\\\"
    done
    echo "\hline"
done

echo "\end{tabular}"
echo "\end{document}"
