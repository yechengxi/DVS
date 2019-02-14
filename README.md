# DVS

CMD:

Latest:  

Using pretrained model:
python main.py /home/cxy/Data/DVS/cloud --pretrained-dispnet pretrained/dispnet_model_best.pth.tar --pretrained-posenet pretrained/exp_pose_model_best.pth.tar -m1 --batch-size 4 -f 50 --lr 1e-3  -s.1 --norm-type fd --sequence-length 5 --slices 25 --sharp --log-output --with-gt --epochs 5


CUDA_VISIBLE_DEVICES=0 python main.py  /home/cxy/Data/DVS/cloud -m.5 --batch-size 8 -f 50 --lr 1e-3  -s0.05  --sequence-length 5  --log-output --norm-type gn --slices 25 --sharp --with-gt  
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  /vulcan/scratch/cxy/Data/DVS/cloud/ -m1 --batch-size 32 -f 50 --lr 1e-3  -s0.1 --norm-type fd --sequence-length 5 --slices 25 --sharp --log-output --with-gt  >seq5.sl25.m1.s.1.fd.log&  


Old:
Training with 4 gpus on vulcan server:

Run inference:

python run_inference.py --img-height 200 --img-width 346 --output-disp --dataset-dir /home/cxy/Data/DVS/MVSEC/outdoor_day_2/eval --pretrained-dispnet pretrained/outdoor/best/dispnet_model_best.pth.tar --pretrained-posenet pretrained/outdoor/best/exp_pose_model_best.pth.tar  --sequence-length 5 --norm-type fd --output-dir /home/cxy/tmp/outdoor_day_2_eval
