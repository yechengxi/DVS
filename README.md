# DVS

CMD:

Training with 4 gpus on vulcan server:

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /vulcan/scratch/cxy/Data/DVS/test/ -m.1 --batch-size 32 --epochs 50 --lr 1e-3 -s0.1 --scale 1 --sequence-length 5 --log-output --simple -o0. --ssim-weight 0. --n-channel 32 --growth-rate 32 --scale-factor 0.5 -f 50  > ecn_32.5.log&

Run inference:

python run_inference.py --img-height 200 --img-width 346 --output-disp --dataset-list outdoor_night1 --dataset-dir /home/cxy/Data/DVS/test/outdoor_night --pretrained-dispnet /home/cxy/Dropbox/Projects/DVS/checkpoints/test,20epochs,seq5,b32,cosine,Adam,lr0.001,m0.1,s0.1,p0.0,o0.0/09-10-19:15/dispnet_model_best.pth.tar --pretrained-posenet /home/cxy/Dropbox/Projects/DVS/checkpoints/test,20epochs,seq5,b32,cosine,Adam,lr0.001,m0.1,s0.1,p0.0,o0.0/09-10-19:15/exp_pose_model_best.pth.tar  --sequence-length 5

