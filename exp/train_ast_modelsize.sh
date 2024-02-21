cd ../train

python train.py --model ast_pytorch --model_size tiny224 --learning_rate 0.001 --is_taper 1 --epoch 200
python train.py --model ast_pytorch --model_size small224 --learning_rate 0.001 --is_taper 1 --epoch 200
python train.py --model ast_pytorch --model_size base224 --learning_rate 0.001 --is_taper 1 --epoch 200
python train.py --model ast_pytorch --model_size base384 --learning_rate 0.001 --is_taper 1 --epoch 200