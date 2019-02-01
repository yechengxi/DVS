# DVS

CMD:

Latest:

data_dir=/home/cxy/Data/DVS/lab/
data_dir=/vulcan/scratch/cxy/Data/DVS/lab/
CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1  --sequence-length 3  --log-output --with-gt --final-map-size 8 -p1 --epochs 200 >lab3.log&

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1 -d.5 --sequence-length 5  --log-output --final-map-size 8 -p2 --epochs 50 -j 16 --pretrained-dispnet checkpoints/dispnet_checkpoint.pth.tar  >candidate2.log&

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 16 -f 50 --lr 1e-4  -s10 -d.0 --sequence-length 3  --log-output --final-map-size 8 -p2 --epochs 50 -j 16 --pretrained-dispnet pretrained_us/dispnet_checkpoint.pth.tar --pretrained-posenet pretrained_us/exp_pose_checkpoint.pth.tar  >unsupervised.log&

CUDA_VISIBLE_DEVICES=0,1 python main.py $data_dir -m1 --batch-size 16 -f 50 --lr 1e-4  -s10 -d.0 --sequence-length 3  --log-output --final-map-size 8 -p2 --epochs 50 -j 16 --pretrained-dispnet pretrained/unsupervise/dispnet_checkpoint.pth.tar --pretrained-posenet pretrained/unsupervise/exp_pose_checkpoint.pth.tar  >unsupervised.log&

data_dir=/vulcan/scratch/cxy/Data/DVS/lab3/
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1 -d.5  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p2 --epochs 50 -j 16 --pretrained-dispnet pretrained_c1/dispnet_checkpoint.pth.tar --pretrained-posenet pretrained_c1/exp_pose_checkpoint.pth.tar --pixelpose >pixelwise_finetune.log&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 2e-4  -s1 -d.5  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p2 --epochs 10 -j 16 --pretrained-dispnet pretrained_c2/dispnet_checkpoint.pth.tar --pretrained-posenet pretrained_c2/exp_pose_checkpoint.pth.tar --lr-scheduler none >mask_finetune.log&



data_dir=/home/cxy/Data/DVS/lab2/CUBE_MEDIUM_PLANE
dispnet_dir=pretrained/pixel3/dispnet_checkpoint.pth.tar
posenet_dir=pretrained/pixel3/exp_pose_checkpoint.pth.tar
output_dir=/home/cxy/CUBE_MEDIUM_PLANE_output_pixel3
CUDA_VISIBLE_DEVICES=0 python run_inference.py --img-height 260 --img-width 346 --final-map-size 8 --output-disp --dataset-dir $data_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir --sequence-length 5 --output-dir $output_dir --pixelpose

cd ~
tar -zcvf CUBE_MEDIUM_PLANE_output_pixel.tar.gz CUBE_MEDIUM_PLANE_output_pixel

data_dir=/home/cxy/Data/DVS/lab2/CUBE_MEDIUM_PLANE
output_dir=/home/cxy/CUBE_MEDIUM_PLANE_output_mask

dispnet_dir=pretrained/mask/dispnet_checkpoint.pth.tar
posenet_dir=pretrained/mask/exp_pose_checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=1 python run_inference.py --img-height 260 --img-width 346 --final-map-size 8 --output-disp --dataset-dir $data_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir --sequence-length 5 --output-dir $output_dir

cd ~
tar -zcvf CUBE_MEDIUM_PLANE_output_mask.tar.gz CUBE_MEDIUM_PLANE_output_mask




cd ~
tar -zcvf CUBE_MEDIUM_PLANE_output_pixel3.tar.gz CUBE_MEDIUM_PLANE_output_pixel3


CUDA_VISIBLE_DEVICES=0  python main.py $data_dir -m1 --batch-size 4 -f 50 --lr 1e-3  -s1  --sequence-length 3  --log-output --with-gt --final-map-size 8 -p1 --epochs 50 -d.0 >unsupervised.log&






data_dir=/home/cxy/Data/DVS/lab3/O1O2O3_SMOOTH-001
output_dir=/home/cxy/O1O2O3_SMOOTH-001_output_mask

dispnet_dir=pretrained/mask/dispnet_checkpoint.pth.tar
posenet_dir=pretrained/mask/exp_pose_checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=1 python run_inference.py --img-height 260 --img-width 346 --final-map-size 8 --output-disp --dataset-dir $data_dir --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir --sequence-length 5 --output-dir $output_dir

cd ~
tar -zcvf CUBE_MEDIUM_PLANE_output_mask.tar.gz CUBE_MEDIUM_PLANE_output_mask




data_dir=/vulcan/scratch/cxy/Data/DVS/lab3/
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1 -d.5  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p2 --epochs 50 -j 16 --pretrained-dispnet pretrained_c1/dispnet_checkpoint.pth.tar --pretrained-posenet pretrained_c1/exp_pose_checkpoint.pth.tar --pixelpose >pixelwise_finetune.log&
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 2e-4  -s1 -d.5  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p2 --epochs 10 -j 16 --pretrained-dispnet pretrained_c2/dispnet_checkpoint.pth.tar --pretrained-posenet pretrained_c2/exp_pose_checkpoint.pth.tar --lr-scheduler none >mask_finetune.log&



data_dir=/home/cxy/Data/DVS/lab2/CUBE_MEDIUM_PLANE
dispnet_dir=pretrained/mask/dispnet_checkpoint.pth.tar
posenet_dir=pretrained/mask/exp_pose_checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=0 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-4  -s1  --sequence-length 5 --slices 25 --log-output --with-gt --final-map-size 8 -p1 --sharp --epochs 20 --pretrained-dispnet $dispnet_dir --pretrained-posenet $posenet_dir >slice.log&



data_dir=/vulcan/scratch/cxy/Data/DVS/lab3/
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-4 -s1 -d.5  --sequence-length 5 --slices 25  --log-output --with-gt --final-map-size 8 -p2 --epochs 50 -j 16 --pretrained-dispnet pretrained_c2/dispnet_checkpoint.pth.tar --pretrained-posenet pretrained_c2/exp_pose_checkpoint.pth.tar --sharp >mask_cloud.log&


data_dir=/vulcan/scratch/cxy/Data/DVS/lab/

CUDA_VISIBLE_DEVICES=0,1  python main.py $data_dir -m1 --batch-size 16 -f 50 --lr 1e-3  -s1  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p1 --epochs 20 --scale-factor 0.3 --n-channel 8 --growth-rate 8 -j 16 >small.log&


CUDA_VISIBLE_DEVICES=2,3  python main.py $data_dir -m1 --batch-size 16 -f 50 --lr 1e-3  -s1.001  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p1 --epochs 20 --scale-factor 0.3 --n-channel 8 --growth-rate 8 -j 16 --norm-type fd >small_fd.log&

CUDA_VISIBLE_DEVICES=0,1  python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1.0001  --sequence-length 5  --log-output --with-gt --final-map-size 4 -p1 --epochs 20 --scale-factor 0.3 --n-channel 4 --growth-rate 4 -j 16 --norm-type fd >small_deep.log&

CUDA_VISIBLE_DEVICES=2,3  python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1.01  --sequence-length 5  --log-output --with-gt --final-map-size 4 -p1 --epochs 20 --scale-factor 0.5 --n-channel 4 --growth-rate 4 -j 16 --norm-type fd >small_deeper.log&

CUDA_VISIBLE_DEVICES=0,1  python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1.001  --sequence-length 5  --log-output --with-gt --final-map-size 4 -p1 --epochs 20 --scale-factor 0.4 --n-channel 4 --growth-rate 4 -j 16 --norm-type fd >small_deeper2.log&



CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 2e-4  -s1 -d.5  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p2 --epochs 20 -j 16 --pretrained-dispnet pretrained_c2/dispnet_checkpoint.pth.tar --pretrained-posenet pretrained_c2/exp_pose_checkpoint.pth.tar --lr-scheduler none --norm-type fd >mask_finetune_fd.log&






CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1 -d.5 --sequence-length 5  --log-output --final-map-size 8 -p2 --epochs 100 -j 16 >mask_model.log&

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 2e-4  -s1 -d.5 --sequence-length 5  --log-output --final-map-size 1 -p2 --epochs 100 -j 16 >mask_model2.log&


data_dir=/vulcan/scratch/cxy/Data/DVS/lab/

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 32 -f 50 --lr 1e-3  -s1 -d.5 --sequence-length 5  --log-output --final-map-size 8 -p2 --epochs 100 -j 16 >mask_model3_sparse_penalty.log&



data_dir=/vulcan/scratch/cxy/Data/DVS/lab/
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1.01 --batch-size 32 -f 50 --lr 2e-4 -s1 -d.5  --sequence-length 5 --slices 0  --log-output --with-gt --final-map-size 8 -p2 --epochs 50 -j 16 --pretrained-dispnet pretrained/mask3/dispnet_checkpoint.pth.tar -c4 >4comp_2.log&




data_dir=/vulcan/scratch/anton/EV-IMO-learning
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 16 -f 500 --lr 1e-3  -s1  --sequence-length 5 --log-output --with-gt --final-map-size 8 -p1 --epochs 20  >no_slice_1.log&

data_dir=/vulcan/scratch/anton/EV-IMO-learning
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 24 -f 500 --lr 1e-3  -s1  --sequence-length 5 --slices 25 --log-output --with-gt --final-map-size 8 -p1 --sharp --epochs 50  >slice_2.log&


data_dir=/vulcan/scratch/anton/EV-IMO-learning
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 64 -f 500 --lr 1e-3  -s1  --sequence-length 5 --slices 5 --log-output --with-gt --final-map-size 8 -p1 --sharp --epochs 50  >slice_2.5.log&


CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1.01 --batch-size 24 -f 100 --lr 1e-3  -s1  --sequence-length 5 --slices 25 --log-output --with-gt --final-map-size 8 -p1 --sharp --epochs 50  >slice_4_new_loader.log&

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1.0 --batch-size 24 -f 100 --lr 1e-3  -s1  --sequence-length 5  --log-output --with-gt --final-map-size 8 -p1 --epochs 50  >no_slice_4.5_new_loader.log&


CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1.01 --batch-size 24 -f 100 --lr 1e-3  -s1  --sequence-length 5 --slices 5 --log-output --with-gt --final-map-size 8 -p1 --sharp --epochs 50  >slice_4.5_new_loader_2.log&



data_dir=/vulcan/scratch/anton/EV-IMO-learning-1
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1.0 --batch-size 16 -f 100 --lr 1e-4  -s1  --sequence-length 5 --slices 25 --log-output --with-gt --final-map-size 8 -p1 --sharp --epochs 50  >slice_4.25_new_loader.log&

data_dir=/vulcan/scratch/anton/EV-IMO-learning
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 24 -f 500 --lr 1e-3  -s1  --sequence-length 5 --slices 25 --log-output --with-gt --final-map-size 8 -p1 --sharp --epochs 50  >slice_3_old_loader.log&

data_dir=/vulcan/scratch/anton/EV-IMO-learning
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py $data_dir -m1 --batch-size 24 -f 500 --lr 1e-3  -s1  --sequence-length 5 --slices 5 --log-output --with-gt --final-map-size 8 -p1 --sharp --epochs 50  >slice_3.5_old_loader.log&

