export DATA_PATH=/fsx/bigcode/bigcode-training/tokenized_stack_no_pii
envsubst < data/train_data_paths.txt > data/train_data_paths.txt.tmp
envsubst < data/valid_data_paths.txt > data/valid_data_paths.txt.tmp
envsubst < data/test_data_paths.txt > data/test_data_paths.txt.tmp
