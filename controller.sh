#! /bin/bash

sim="0.5"

for var in $sim
do
    echo $var
    python3 main.py --dataset 'cifar10_hard_w10' --n_node 10 --b_rate 0.7 --sim_th $var --net 'vgg11' --gpu
done