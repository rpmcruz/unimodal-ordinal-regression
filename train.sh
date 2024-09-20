#!/bin/bash

DATASETS="HCI ICIAR FGNET SMEAR2005 FOCUSPATH ABALONE5 ABALONE10 BALANCE_SCALE CAR NEW_THYROID"
LOSSES="CrossEntropy POM OrdinalEncoding CrossEntropy_UR CDW_CE BinomialUnimodal_CE PoissonUnimodal UnimodalNet ORD_ACL VS_SL"
LOSSES_LAMBDA="WassersteinUnimodal_KLDIV WassersteinUnimodal_Wass CO2 HO2"
LAMDAS="0.001 0.01 0.1 1 10 100 1000"
REPS=`seq 1 4`

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        # train also split=0 because we use it to produce probabilities figure
        NAME="model-$DATASET-$LOSS-0.pth"
        echo $NAME
        if [ ! -f $NAME ]; then
            python3 train.py $DATASET $LOSS 0 $NAME &
        fi

        for REP in $REPS; do
            NAME="model-$DATASET-$LOSS-$REP.pth"
            echo $NAME
            if [ ! -f $NAME ]; then
                python3 train.py $DATASET $LOSS $REP $NAME &
            fi
        done
        wait
    done
    for LOSS in $LOSSES_LAMBDA; do
        LAMDAS="0.001 0.01 0.1"
        for LAMDA in $LAMDAS; do
            NAME="model-$DATASET-$LOSS-0-lambda-$LAMDA.pth"
            echo $NAME
            if [ ! -f $NAME ]; then
                python3 train.py $DATASET $LOSS 0 $NAME --lamda $LAMDA &
            fi
        done
        wait
        LAMDAS="1 10 100 1000"
        for LAMDA in $LAMDAS; do
            NAME="model-$DATASET-$LOSS-0-lambda-$LAMDA.pth"
            echo $NAME
            if [ ! -f $NAME ]; then
                python3 train.py $DATASET $LOSS 0 $NAME --lamda $LAMDA &
            fi
        done
        wait

        LAMDA=`python3 test-best-lambda.py $DATASET $LOSS`
        for REP in $REPS; do
            NAME="model-$DATASET-$LOSS-$REP-lambda-$LAMDA.pth"
            echo $NAME
            if [ ! -f $NAME ]; then
                python3 train.py $DATASET $LOSS $REP $NAME --lamda $LAMDA &
            fi
        done
        wait
    done
done
