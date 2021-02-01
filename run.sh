#!/bin/bash
for var in {0..63} 
do
    python ./strongLens_rate_cosmos.py 64 $var > output/$var.log &
done
