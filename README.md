# DVS

CMD:

Latest:

data_dir=/home/cxy/Data/DVS/lab2/
data_dir=/vulcan/scratch/cxy/Data/DVS/lab2/
CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1  --sequence-length 3  --log-output --with-gt --final-map-size 8 -p1 --epochs 200 >lab3.log&

CUDA_VISIBLE_DEVICES=1 python run_inference.py --img-height 260 --img-width 346 --final-map-size 8 --output-disp --dataset-dir /home/cxy/Data/DVS/lab2/EASY_CAR_PLANE2 --pretrained-dispnet checkpoints/dispnet_checkpoint.pth.tar --pretrained-posenet checkpoints/exp_pose_checkpoint.pth.tar   --sequence-length 3 --output-dir /home/cxy/output2
