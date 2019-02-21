
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math

from matplotlib.colors import LogNorm


def visualize_all_maps(out,msg):
    im_idx=0
    channels=[out[i].shape[1] for i in range(len(out))]

    growth_rate=out[-1].shape[1]-out[-2].shape[1]
    max_c=8
    n_layers=len(out)

    for layer in range(1,1+n_layers):
        cols=max(channels[:layer])
        cnt=0
        for level in range(0,cols,growth_rate):
            for c in range(max_c):
                if level+c<cols:
                    cnt+=1

        #fig = plt.figure(figsize=(cnt*5, 5))
        fig = plt.figure(figsize=(min(cnt,max_c) * 5, math.ceil(cnt/max_c) *3))

        #gs = gridspec.GridSpec(1, cnt, wspace=0.05, hspace=0.05)
        gs = gridspec.GridSpec(math.ceil(cnt/max_c), min(cnt,max_c), wspace=0.05, hspace=0.05)

        ims=out[layer-1]
        cnt = 0
        r=0
        for level in range(0, cols, growth_rate):

            for c in range(max_c):
                if level + c < cols:
                    #ax = plt.subplot(gs[0, cnt])
                    #print(r,c)
                    ax = plt.subplot(gs[r, c])
                    img=ims[im_idx, cnt].cpu().data.numpy()
                    img = (img - img.mean()) / (img.std()+1e-6)
                    img[img<-3]=-3
                    img[img > 3]=3
                    #img=(img-img.min())/(img.max()-img.min()+1e-6)
                    #img=img.__pow__(0.5)
                    ax.imshow(img, cmap='gray')
                    ax.set_axis_off()
                    cnt = cnt + 1

            else:
                None#ax.set_visible(False)

            r+=1
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.savefig(msg + '_maps_layer%d'%(layer), bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close('all')


