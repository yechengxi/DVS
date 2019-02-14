# DVS

CMD:

Latest:

Training with 4 gpus on vulcan server:

data_dir=/home/cxy/Data/DVS/MVSEC
data_dir=/vulcan/scratch/cxy/Data/DVS/MVSEC/

1. full ecn with fd
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output --simple  --with-gt  --norm-type fd --final-map-size 4 --epochs 50 >seq5.fd_normalized.log&
2. tiny ecn with fd
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output --simple  --with-gt  --norm-type fd --n-channel 8 --growth-rate 8 --final-map-size 4 --epochs 50 >seq5.fd.tiny.log&
3. ecn with bn
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output --simple  --with-gt  --norm-type bn --final-map-size 4 --epochs 50 >seq5.bn.log&
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-3  --sequence-length 5 --log-output --simple  --with-gt  --norm-type bn --final-map-size 4 --epochs 50 >seq5.bn2.log&
4. ecn with gn
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-2  --sequence-length 5 --log-output --simple  --with-gt  --norm-type gn --final-map-size 4 --epochs 50 >seq5.gn.log&
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $data_dir -j 32 -m.1 --batch-size 32 -f 50 --lr 1e-3  --sequence-length 5 --log-output --simple  --with-gt  --norm-type gn --final-map-size 4 --epochs 50 >seq5.gn2.log&



Run inference:

dataset_dir=/vulcan/scratch/cxy/Data/DVS/MVSEC/outdoor_night_1/eval
output_dir=./results/outdoor_night_1/eval


dispnet_dir=pretrained/MVSEC/tiny/dispnet_model_best.pth.tar
posenet_dir=pretrained/MVSEC/tiny/exp_pose_model_best.pth.tar
python run_inference.py --output-disp --dataset-dir $dataset_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir  --sequence-length 5 --norm-type fd --output-dir $output_dir --n-channel 8 --growth-rate 8 --final-map-size 4



dispnet_dir=pretrained/MVSEC/ecn_fd/dispnet_model_best.pth.tar
posenet_dir=pretrained/MVSEC/ecn_fd/exp_pose_model_best.pth.tar

python run_inference.py --output-disp --dataset-dir $dataset_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir  --sequence-length 5 --norm-type fd --output-dir $output_dir --final-map-size 4



dispnet_dir=pretrained/MVSEC/sfmlearner/dispnet_model_best.pth.tar
posenet_dir=pretrained/MVSEC/sfmlearner/exp_pose_model_best.pth.tar

python run_inference.py --output-disp --dataset-dir $dataset_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir  --sequence-length 5 --output-dir $output_dir --arch std 




