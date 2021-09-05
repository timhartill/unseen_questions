# run sklearn classifiers against selected datasets

cd ../code

python calc_accuracy_sklearn.py --output_dir /data/thar011/out/unifiedqa_averages/classifier_results \
        --print_report --print_cm --input_dir /data/thar011/data/unifiedqa/strategy_qa

python calc_accuracy_sklearn.py --output_dir /data/thar011/out/unifiedqa_averages/classifier_results \
        --print_report --print_cm --input_dir /data/thar011/data/jiant_combined_datasets/pararules_depth-3ext-NatLang

python calc_accuracy_sklearn.py --output_dir /data/thar011/out/unifiedqa_averages/classifier_results \
        --print_report --print_cm --input_dir /data/thar011/data/jiant_combined_datasets/conceptrules_v2_simplified

python calc_accuracy_sklearn.py --output_dir /data/thar011/out/unifiedqa_averages/classifier_results \
        --print_report --print_cm --input_dir /data/thar011/data/jiant_combined_datasets/conceptrules_v2_full



