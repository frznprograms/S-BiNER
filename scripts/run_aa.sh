#!/bin/bash

DATA_FILE=../data/cleaned_data/aa_df.csv
MODEL_NAME_OR_PATH=../models/awesome-align/
OUTPUT_FILE=../output/awesome_alignments.txt

awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --batch_size 32 \
    --num_workers 0