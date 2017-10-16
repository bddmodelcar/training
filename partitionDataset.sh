#!/bin/bash

sym_data () {
  i=$((0))
  while read line
  do
    echo /hostroot$line
    if (($i<9))
    then
      ln -s /hostroot$line ./data/val/processed_h5py/$i
    else
      ln -s /hostroot$line ./data/train/processed_h5py/$i
    fi
    i=$((i+1))
  done
}
shuf -e /data/dataset/bair_car_data_Main_Dataset/processed_h5py/* | sym_data
