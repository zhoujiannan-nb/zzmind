#!/bin/bash
cd /mnt/boot/datasets/zzmind/data_aug
rm -f STOP
nohup python3 gen.py run >> full.out 2>&1 &
echo "started pid $!"
