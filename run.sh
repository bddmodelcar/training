#!/bin/bash
set -x
set -e

for ((i=0; i<=1000; i++)); do
    bdd-docker python Train.py --epoch $i "$@" 
done
