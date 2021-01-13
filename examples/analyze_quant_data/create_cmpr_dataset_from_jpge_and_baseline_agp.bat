@cls
python calculate_baseline_data.py ^
  --input-conf-file .\confs\bsln_conf.yaml

python calculate_jpeg_data.py ^
  --input-conf-file .\confs\jpeg_conf.yaml

python calculate_raw_agp_data.py ^
  --input-conf-file .\confs\agp_conf.yaml
