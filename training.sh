GPUID=1

# DUET
#python train_AUTE.py --gpu $GPUID --model-num 3 --num-class 10 --steps 6 --eps 0.0314 \
#--alpha 0.008 --batch-size 256 --arch "ResNet20" --beta 1.0 --margin 0.01 --epochs 120

python train_AUTE.py --gpu $GPUID --model-num 3 --num-class 100 --steps 6 --eps 0.0314 \
--alpha 0.008 --batch-size 256 --arch "ResNet18" --beta 3.0 --margin 0.6 --epochs 120

# AROW
#python train/train_arow.py --gpu $GPUID --model-num 1 --num-class 100 --steps 10 --eps 0.031 \
#--alpha 0.008 --batch-size 128 --arch "WideResNet" --depth 18 --gamma 6.0

# ADV
#python train/train_adv.py --gpu $GPUID --model-num 3 --num-class 10 --steps 10 --eps 0.031 \
#--alpha 0.008 --batch-size 128 --arch "WideResNet" --depth 34

# TRADES
#python train/train_trades.py --gpu $GPUID --model-num 3 --num-class 10 --steps 10 --eps 0.031 \
#--alpha 0.008 --batch-size 64 --arch "WideResNet" --depth 34

# MART
#python train/train_mart.py --gpu $GPUID --model-num 3 --num-class 10 --steps 10 --eps 0.031 \
#--alpha 0.008 --batch-size 128 --arch "WideResNet" --depth 34

# HAT
#python train/train_hat2.py --gpu $GPUID --model-num 3 --num-class 10 --steps 10 --eps 0.031 \
#--alpha 0.008 --batch-size 128 --arch "WideResNet" --depth 34

# STAT
#python train/train_stat2.py --gpu $GPUID --model-num 3 --num-class 10 --steps 10 --eps 0.031 \
#--alpha 0.008 --batch-size 64 --arch "WideResNet" --depth 34
