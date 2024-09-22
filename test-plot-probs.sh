#!/bin/bash

DATASET=FGNET
LOSSES="CrossEntropy POM OrdinalEncoding CDW_CE BinomialUnimodal_CE PoissonUnimodal ORD_ACL VS_SL UnimodalNet CrossEntropy_UR"
LOSSES_LAMBDA="CO2 WassersteinUnimodal_KLDIV WassersteinUnimodal_Wass"

MODELS=()
for LOSS in $LOSSES; do
  MODELS+=("model-$DATASET-${LOSS}-0.pth")
done
for LOSS in $LOSSES_LAMBDA; do
  LAMBDA=`python3 test-best-lambda.py $DATASET $LOSS`
  MODELS+=("model-$DATASET-${LOSS}-0-lambda-${LAMBDA}.pth")
done

python3 test-plot-probs.py "${MODELS[@]}" --dataset $DATASET
