# DVS

CMD:

Latest:

data_dir=/vulcan/scratch/cxy/Data/DVS/lab/

CUDA_VISIBLE_DEVICES=1 python main.py $data_dir -m1 --batch-size 16 -f 50 --lr 1e-3  -s1  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p1 --epochs 200 >lab1.log&
