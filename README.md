# DVS

CMD:

Latest:

data_dir=/home/cxy/Data/DVS/lab2/
data_dir=/vulcan/scratch/cxy/Data/DVS/lab2/
CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p1 --epochs 200 >lab3.log&

