cd ../train

python train.py --model ast_pytorch --model_size base224 --learning_rate 0.001 --is_taper 1 --epoch 200 --ft_stride 3
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.001 --is_taper 1 --epoch 200 --ft_stride 5
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.001 --is_taper 1 --epoch 200 --ft_stride 7
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.001 --is_taper 1 --epoch 200 --ft_stride 9
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.001 --is_taper 1 --epoch 200 --ft_stride 11
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.001 --is_taper 1 --epoch 200 --ft_stride 13