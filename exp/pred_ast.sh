cd ../postprocess

python pred.py --model ast_pytorch --model_size base224 --learning_rate 0.00005 --is_taper 1 --epoch 200 --ft_stride 5 --time_str 2024-02-21_02-14
python pred.py --model ast_pytorch --model_size base224 --learning_rate 0.00005 --is_taper 0 --epoch 200 --ft_stride 5 --time_str 2024-02-21_13-56