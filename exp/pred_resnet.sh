cd ../postprocess

python pred.py --model resnet --learning_rate 0.0001 --is_taper 0 --epoch 200 --time_str 2024-02-21_21-31
python pred.py --model resnet --learning_rate 0.0001 --is_taper 1 --epoch 200 --time_str 2024-02-21_23-31