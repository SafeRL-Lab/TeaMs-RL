#!/bin/bash

while true; do
    python generate_data_alpaca.py --algo 2SR --seed 543 --cost-limit 0.7 --exps-epoch 41000 #26000 # 52002    
    
    exit_status=$?
    
    if [ $exit_status -eq 0 ]; then
        echo "Python program exited normally."
    else
        echo "Python program exited with error. Restarting..."
    fi    
    
    sleep 5
done
