#!/bin/bash

DATASET=FGNET
LOSSES="CrossEntropy POM OrdinalEncoding CrossEntropy_UR CDW_CE BinomialUnimodal_CE PoissonUnimodal UnimodalNet"
LOSSES_LAMBDA="CO2 HO2 WassersteinUnimodal_KLDIV WassersteinUnimodal_Wass"

MODELS=()
for LOSS in $LOSSES; do
  MODELS+=("model-$DATASET-${LOSS}-0.pth")
done
for LOSS in $LOSSES_LAMBDA; do
  LAMBDA=`python3 test-best-lambda.py $DATASET $LOSS`
  MODELS+=("model-$DATASET-${LOSS}-0-lambda-${LAMBDA}.pth")
done

python3 test-plot-probs.py $DATASET quantiles 4 --models "${MODELS[@]}"
