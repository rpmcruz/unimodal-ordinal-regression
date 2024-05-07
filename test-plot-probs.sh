#!/bin/bash
DATASETS="ICIAR HCI FGNET SMEAR2005 FOCUSPATH"
AGGS="avg all"
KLASSES="first second middle last"
for DATASET in $DATASETS; do
for AGG in $AGGS; do
for KLASS in $KLASSES; do
python3 show_probs.py $DATASET $AGG $KLASS
done
done
done
