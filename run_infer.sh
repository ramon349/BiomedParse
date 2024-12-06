output_dir="../biomedParseMonai/output_dir"
inference_maps="../biomedParseMonai/inference_maps"
python3 ./ramon_infer.py --data_path "../data/sam_dataset.pkl" --output_dir "${output_dir}" --name "SamData" --inference_map_path "${inference_maps}"
