GPUID=4
arch='ResNet20'
path=/nfs/AUTE/results/AUTE/seed_0/3_ResNet20/eps0.031_steps6/epoch_120.pth
num=1000

python eval/eval_mim.py \
    --gpu $GPUID \
    --model-file $path \
    --steps 50 \
    --num-class 10 \
    --arch $arch --depth 20 \
    --test-size $num \
    --random-start 1 \
    --save-to-csv

python eval/eval_wbox.py \
  --gpu $GPUID \
  --model-file $path \
  --steps 10 \
  --random-start 1 \
  --num-class 10 \
  --test-size $num \
  --arch $arch --depth 20 \
  --save-to-csv

python eval/eval_wbox.py \
  --gpu $GPUID \
  --model-file $path \
  --steps 100 \
  --random-start 1 \
  --num-class 10 \
  --test-size $num \
  --arch $arch --depth 20 \
  --save-to-csv

python eval/eval_wbox_aa.py \
    --gpu $GPUID \
    --model-file $path \
    --steps 50 \
    --random-start 1 \
    --arch $arch --depth 20 \
    --test-size $num \
    --num-class 10 \
    --save-to-csv

