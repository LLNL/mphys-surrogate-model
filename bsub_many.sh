#!/bin/bash

declare -a arr=("1" "2" "3" "4" "5" "6" "7" "8")

for num in "${arr[@]}"
do
    bsub submit.bsub
done 
