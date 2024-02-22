cd ../train
python train.py --model resnet --learning_rate 0.06 --is_taper 0 --epoch 200
python train.py --model resnet --learning_rate 0.06 --is_taper 1 --epoch 200
python train.py --model resnet --learning_rate 0.07 --is_taper 0 --epoch 200
python train.py --model resnet --learning_rate 0.07 --is_taper 1 --epoch 200
python train.py --model resnet --learning_rate 0.08 --is_taper 0 --epoch 200
python train.py --model resnet --learning_rate 0.08 --is_taper 1 --epoch 200