# DVS

CMD:

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /vulcan/scratch/cxy/Data/DVS/test/ -m.1 --batch-size 32 -f 50 --lr 1e-3 -s0.1 --scale 1 --sequence-length 5 --log-output --simple -o0. --ssim-weight 0. --n-channel 32 --growth-rate 32 --scale-factor 0.5 > ecn_32.5.log&
