# DVS

CMD:

Latest:

python main.py /home/cxy/Data/DVS/od1 -m.1 --batch-size 8 -f 50 --lr 1e-3  -s0.01 --scale 1. --sequence-length 5 --log-output --simple -o0. --ssim-weight 0. --norm-type gn

Training with 4 gpus on vulcan server:

data_dir=/home/cxy/Data/DVS/MVSEC
data_dir=/vulcan/scratch/cxy/Data/DVS/MVSEC/


CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 16 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output --simple  --with-gt  --norm-type fd --final-map-size 4 >seq5.s.1.fd.log&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 16 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output --simple  --with-gt  --norm-type fd --n-channel 8 --growth-rate 8 --final-map-size 4 >seq5.s.1.fd.tiny.log&
python main.py $data_dir -j 16 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output --simple  --with-gt  --norm-type fd --n-channel 8 --growth-rate 8 --final-map-size 8 

Run inference:
dataset_dir=/home/cxy/Data/DVS/MVSEC/outdoor_night_1/eval
dispnet_dir=pretrained/outdoor/old/dispnet_model_best.pth.tar
posenet_dir=pretrained/outdoor/old/exp_pose_model_best.pth.tar
output_dir=~/results/outdoor_night_1/eval
python run_inference.py --img-height 200 --img-width 346 --output-disp --dataset-dir $dataset_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir  --sequence-length 5 --norm-type fd --output-dir $output_dir


