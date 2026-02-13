#!/bin/bash


#for lr in 0.0001 0.00005 0.0002
#do
#  for batch in 256 512
#  do
#    python train_sar_ships.py --lr $lr --batch $batch --name "lr${lr}_b${batch}"
#  done
#done
#



#!/bin/bash
for lr in 0.00015 0.0002; do
  for batch in 512 768; do
    for nms in 0.4; do
      for score in 0.4 0.5; do
        name="v5_lr${lr}_b${batch}_nms${nms}_score${score}"
        echo " Running: $name"
        python3 trainv4_Ships_sweeps_v2.py \
          --lr $lr \
          --batch $batch \
          --nms $nms \
          --score $score \
          --name $name
      done
    done
  done
done
