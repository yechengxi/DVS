# DVS
main.py is the training protocol. 
run_inference.py runs the inference part once you have a pretrained model.


CMD:

Latest:

Training with 4 gpus on vulcan server:

data_dir=/home/cxy/Data/DVS/MVSEC
data_dir=/vulcan/scratch/cxy/Data/DVS/MVSEC/

1. full ecn with fd
CUDA_VISIBLE_DEVICES=0,1 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --final-map-size 4 --epochs 50 >seq5.fd.log&
2. tiny ecn with fd
CUDA_VISIBLE_DEVICES=2,3 python main.py $data_dir -j 32 -m.101 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --n-channel 8 --growth-rate 8 --final-map-size 4 --epochs 50 >seq5.fd.tiny.log&
3. ecn with bn
CUDA_VISIBLE_DEVICES=4,5 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type bn --final-map-size 4 --epochs 50 >seq5.bn.log&
4. ecn with gn
CUDA_VISIBLE_DEVICES=6,7 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type gn --final-map-size 4 --epochs 50 >seq5.gn.log&


data_dir=/vulcan/scratch/cxy/Data/DVS/MVSEC/

1. tiny ecn with fd
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $data_dir -j 32 -m.1001 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd  --epochs 30 >outdoor.fd.log&
2. tiny ecn with bn
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.1001 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type bn --epochs 30 >outdoor.bn.log&
3. tiny ecn with gn
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $data_dir -j 32 -m.1001 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type gn --epochs 30 >outdoor.gn.log&

4. fd g8
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.1001 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --norm-group 8 --epochs 30 >outdoor.fd.g8.log&


###outdoor day2
data_dir=/vulcan/scratch/cxy/Data/DVS/MVSEC/outdoor_day_2/



CUDA_VISIBLE_DEVICES=6,7 python main.py $data_dir -j 16 -m.1 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 3 --log-output  --with-gt  --norm-type fd --final-map-size 4 --epochs 20 >outdoor_day2.fd.seq3.log&

CUDA_VISIBLE_DEVICES=2,3 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 7 --log-output  --with-gt  --norm-type fd --final-map-size 4 --epochs 20 >outdoor_day2.fd.seq7.log&

CUDA_VISIBLE_DEVICES=4,5 python main.py $data_dir -j 16 -m.1 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 9 --log-output  --with-gt  --norm-type fd --final-map-size 4 --epochs 20 >outdoor_day2.fd.seq9.log&



1. tiny ecn with fd
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 5 --log-output  --with-gt --norm-type fd --epochs 30 >fd.log&
2. tiny ecn with bn
CUDA_VISIBLE_DEVICES=4,5 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 5 --log-output --with-gt  --norm-type bn --epochs 30 >bn.log&
3. tiny ecn with gn
CUDA_VISIBLE_DEVICES=6,7 python main.py $data_dir -j 32 -m.1 --batch-size 16 -f 100 --lr 1e-2  --sequence-length 5 --log-output --with-gt  --norm-type gn --epochs 30 >gn.log&
4. fd g8
CUDA_VISIBLE_DEVICES=0,1 python main.py $data_dir -j 32 -m.101 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --norm-group 8 --epochs 30 >fd.g8.log&


CUDA_VISIBLE_DEVICES=2,3 python main.py $data_dir -j 32 -m.101 --batch-size 32 -f 100 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd_v3 --epochs 30 >fd_v3.log&



Run inference:

dataset_dir=/vulcan/scratch/cxy/Data/DVS/MVSEC/outdoor_day_2/eval
output_dir=./results/outdoor_night_1/eval


dispnet_dir=pretrained/MVSEC/tiny/dispnet_model_best.pth.tar
posenet_dir=pretrained/MVSEC/tiny/exp_pose_model_best.pth.tar
python run_inference.py --dataset-dir $dataset_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir  --sequence-length 5 --norm-type fd --output-dir $output_dir --n-channel 8 --growth-rate 8 --final-map-size 4



dispnet_dir=pretrained/MVSEC/ecn_fd/dispnet_model_best.pth.tar
posenet_dir=pretrained/MVSEC/ecn_fd/exp_pose_model_best.pth.tar

python run_inference.py --dataset-dir $dataset_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir  --sequence-length 5 --norm-type fd --output-dir $output_dir --final-map-size 4



dispnet_dir=pretrained/MVSEC/sfmlearner/dispnet_model_best.pth.tar
posenet_dir=pretrained/MVSEC/sfmlearner/exp_pose_model_best.pth.tar

python run_inference.py --dataset-dir $dataset_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir  --sequence-length 5 --output-dir $output_dir --arch std 




dispnet_dir=pretrained/MVSEC/bn/dispnet_model_best.pth.tar
posenet_dir=pretrained/MVSEC/bn/exp_pose_model_best.pth.tar

CUDA_VISIBLE_DEVICES=0 python run_inference.py --dataset-dir $dataset_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir  --sequence-length 5 --norm-type bn --output-dir $output_dir --final-map-size 4


##indoor
data_dir=/vulcan/scratch/cxy/Data/DVS/indoor
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --final-map-size 4 --epochs 50 >indoor.log&
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $data_dir -j 32 -m.101 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --n-channel 8 --growth-rate 8 --final-map-size 4 --epochs 50 >indoor.tiny.log&

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.101 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --n-channel 4 --growth-rate 4 --final-map-size 8 --epochs 50 >indoor.super.tiny.log&

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $data_dir -j 32 -m.103 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --n-channel 4 --growth-rate 4 --final-map-size 8 --scale-factor .3 --epochs 50 >indoor.super.super.tiny.log&


##tiny things
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $data_dir -j 32 -m.102 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --n-channel 4 --growth-rate 4 --final-map-size 8 --epochs 50 >outdoor.super.tiny.log&

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.103 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --n-channel 4 --growth-rate 4 --final-map-size 8 --scale-factor .3 --epochs 50 >outdoor.super.super.tiny.log&

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.102 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output  --with-gt  --norm-type fd --norm-group 8 --n-channel 4 --growth-rate 4 --final-map-size 8 --epochs 50 >outdoor.super.tiny.g8.log&
