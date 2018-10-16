# DVS

CMD:

Latest:  

python main.py /home/cxy/Data/DVS/cloud/ --batch-size 4 -f 50 --lr 1e-3  -s0.05 --sequence-length 25 --log-output --sharp  --norm-type gn


Old:
Training with 4 gpus on vulcan server:

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /vulcan/scratch/cxy/Data/DVS/test/ -m.1 --batch-size 32 --epochs 50 --lr 1e-3 -s0.1 --scale 1 --sequence-length 5 --log-output --simple -o0. --ssim-weight 0. --n-channel 32 --growth-rate 32 --scale-factor 0.5 -f 50  > ecn_32.5.log&

Run inference:

python run_inference.py --img-height 200 --img-width 346 --output-disp --dataset-dir /home/cxy/Data/DVS/MVSEC/outdoor_day_2/eval --pretrained-dispnet pretrained/outdoor/best/dispnet_model_best.pth.tar --pretrained-posenet pretrained/outdoor/best/exp_pose_model_best.pth.tar  --sequence-length 5 --norm-type fd --output-dir /home/cxy/tmp/outdoor_day_2_eval
