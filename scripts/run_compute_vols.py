#!/bin/bash
python src/compute_all_vols.py\
    --input_dir data/filtered_data\
    --output_file output/all_vols.csv\
    --intervals 1min 5min 15min 30min\
    --estimator std
