cd ../train
#python train.py --model ast_pytorch --model_size base224 --learning_rate 0.05 --is_taper 1 --epoch 200 --ft_stride 5
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.01 --is_taper 1 --epoch 200 --ft_stride 5
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.005 --is_taper 1 --epoch 200 --ft_stride 5
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.001 --is_taper 1 --epoch 200 --ft_stride 5
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.0005 --is_taper 1 --epoch 200 --ft_stride 5
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.0001 --is_taper 1 --epoch 200 --ft_stride 5
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.00005 --is_taper 1 --epoch 200 --ft_stride 5
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.00001 --is_taper 1 --epoch 200 --ft_stride 5