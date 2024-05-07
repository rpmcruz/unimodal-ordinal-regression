#!/bin/bash

DATASETS="HCI ICIAR FGNET SMEAR2005 FOCUSPATH ABALONE5 ABALONE10 BALANCE_SCALE CAR NEW_THYROID"
LOSSES="CrossEntropy POM OrdinalEncoding CrossEntropy_UR CDW_CE BinomialUnimodal_CE PoissonUnimodal UnimodalNet"
LOSSES_LAMBDA="WassersteinUnimodal_KLDIV WassersteinUnimodal_Wass CO2 HO2"
LOSSES_LAMBDA=""
LAMDAS="0.001 0.01 0.1 1 10 100 1000"
REPS=`seq 1 4`

if [ $2 -eq "lambda" ]; then
LOSSES_LAMBDA="$3"
else
LOSSES="$3"
fi

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        for REP in $REPS; do
            echo "python3 train.py $DATASET $LOSS $REP"
            python3 train.py $DATASET $LOSS $REP &
        done
        wait
    done
    for LOSS in $LOSSES_LAMBDA; do
        LAMDAS="0.001 0.01 0.1"
        for LAMDA in $LAMDAS; do
            echo "python3 train.py $DATASET $LOSS 0 --lamda $LAMDA"
            python3 train.py $DATASET $LOSS 0 --lamda $LAMDA &
        done
        wait
        LAMDAS="1 10 100 1000"
        for LAMDA in $LAMDAS; do
            echo "python3 train.py $DATASET $LOSS 0 --lamda $LAMDA"
            python3 train.py $DATASET $LOSS 0 --lamda $LAMDA &
        done
        wait

        LAMBDA=`python3 test-best-lambda.py $DATASET $LOSS`
        for REP in $REPS; do
            echo "python3 train.py $DATASET $LOSS $REP --lamda $LAMBDA"
            python3 train.py $DATASET $LOSS $REP --lamda $LAMBDA &
        done
        wait
    done
done
