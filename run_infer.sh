output_dir="../biomedParseMonai/output_dir"
inference_maps="../biomedParseMonai/inference_maps"
python3 ./ramon_infer.py --data_path "../data/kits23_data.pkl" --output_dir "${output_dir}" --name "Kits23" --inference_map_path "${inference_maps}"