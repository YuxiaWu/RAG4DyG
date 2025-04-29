#!/bin/bash

dataset="UCI_13"
python_file="csv2resources.py"
for timestamp in {12..12}
do
    python "$python_file" "$dataset" "$timestamp"
done

dataset="dialog"
for timestamp in {15..15}
do
    python "$python_file" "$dataset" "$timestamp"
done

dataset="hepth" 
for timestamp in {11..11}
do
    python "$python_file" "$dataset" "$timestamp"
done

dataset="enron"
python_file="csv2resources.py"
timestamp = 16
python "$python_file" "$dataset" "$timestamp"


dataset="reddit"
python_file="csv2resources.py"
timestamp = 11
python "$python_file" "$dataset" "$timestamp"

dataset="wikiv2"
python_file="csv2resources.py"
timestamp = 15
python "$python_file" "$dataset" "$timestamp"

