#!/bin/bash

DATASETS="ABALONE5 ABALONE10 BALANCE_SCALE CAR NEW_THYROID HCI ICIAR FGNET SMEAR2005 FOCUSPATH"
LOSSES="CrossEntropy POM OrdinalEncoding CrossEntropy_UR CDW_CE BinomialUnimodal_CE PoissonUnimodal UnimodalNet"
LOSSES_LAMBDA="WassersteinUnimodal_KLDIV WassersteinUnimodal_Wass CO2 HO2"
LAMDAS="0.001 0.01 0.1 1 10 100 1000"
REPS=`seq 1 4`

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        for REP in $REPS; do
            echo "python3 train.py $DATASET $LOSS $REP"
            python3 train.py $DATASET $LOSS $REP
        done
    done
    for LOSS in $LOSSES_LAMBDA; do
        for LAMDA in $LAMDAS; do
            echo "python3 train.py $DATASET $LOSS 0 --lamda $LAMDA"
            python3 train.py $DATASET $LOSS 0 --lamda $LAMDA
        done
        LAMBDA=`python3 test-best-lambda.py $DATASET $LOSS --datadir ../data`
        for REP in $REPS; do
            echo "python3 train.py $DATASET $LOSS $REP --lamda $LAMBDA"
            python3 train.py $DATASET $LOSS $REP --lamda $LAMBDA
        done
    done
done
