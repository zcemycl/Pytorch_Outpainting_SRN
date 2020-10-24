import cv2
from model.loss import *
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms,utils,models
from argparse import Namespace
import matplotlib.pyplot as plt
import pdb
import scipy.stats as st
import scipy

def compressTensor(x):
    return x.cpu().detach().numpy().squeeze()

args = Namespace(
    random_mask=1,
    img_shapes=[256,256,3],
    margins=[0,0],
    mask_shapes=[128,128],
    max_delta_shapes=[0,0],
    feat_expansion_op='subpixel',
    g_cnum=64,
    d_cnum=64,
    use_cn=1,
    fa_alpha=.5,
    pretrain_l1_alpha=1.2,
    wgan_gp_lambda=10,
    gan_loss_alpha=1e-3,
    mrf_alpha=.05,
    l1_loss_alpha=4.2,
    ae_loss_alpha=1.2,
)

class Margin:
    def __init__(self,top=0,left=0,bottom=0,right=0):
        self.top,self.left = top,left
        self.bottom,self.right = bottom,right

def random_square(config):
    img_shape = config.img_shapes
    img_height,img_width = img_shape[:2]
    # random_mask == 1
    maxt = img_height-config.margins[0]-config.mask_shapes[0]   
    maxl = img_width-config.margins[1]-config.mask_shapes[1]
    t = int((config.margins[0]-maxt-1)*torch.rand(1)+maxt+1)
    l = int((config.margins[1]-maxl-1)*torch.rand(1)+maxl+1)

    h = torch.tensor([config.mask_shapes[0]])
    w = torch.tensor([config.mask_shapes[1]])
    margin=Margin(t,l,img_height-config.mask_shapes[0]-t,img_width-config.mask_shapes[1]-l)
    
    return (t,l,h,w),margin

def bbox2mask(bbox,config):
    def npmask(bbox,height,width,delta_h,delta_w):
        mask = np.zeros((1,1,height,width),np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:,:,bbox[0]+h:bbox[0]+bbox[2]-h,bbox[1]+w:bbox[1]+bbox[3]-w] = 1
        return mask
    img_shape = config.img_shapes
    height,width = img_shape[0],img_shape[1]
    mask = npmask(bbox,height,width,config.max_delta_shapes[0],
                    config.max_delta_shapes[1])
    return mask

def gauss_kernel(size=21,sigma=3,inchannels=3,outchannels=3):
    interval = (2*sigma+1)/size
    x = np.linspace(-sigma-interval/2,sigma+interval/2,size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1,1,size,size))
    return out_filter

def make_gauss_var(size,sigma,inchannels=1,outchannels=1):
    kernel = gauss_kernel(size,sigma,inchannels,outchannels)
    var = torch.tensor(kernel)
    return var

def relative_spatial_variant_mask(mask,hsize=64,sigma=1/40,iters=9):
    eps = 1e-5
    kernel = make_gauss_var(hsize,sigma).cuda()
    init = 1-mask

    lay = nn.Conv2d(1,1,64,1,32,bias=False)
    lay.weight = nn.Parameter(kernel,requires_grad=False)
    for i in range(iters):
        #mask_priority = F.pad(lay(torch.tensor(init)),(0,1,0,1))
        mask_priority = lay(init)[:,:,:-1,:-1]
        mask_priority *= mask
        if i == iters-2:
            mask_priority_pre = mask_priority
        init = mask_priority + (1-mask)
    mask_priority = mask_priority_pre/(mask_priority+eps)
    
    #plt.imshow(compressTensor(mask_priority)); plt.show()
    return mask_priority 



class build_generator(nn.Module):
    def __init__(self,config):
        super(build_generator,self).__init__()
        self.config = config
        self._init_generator()
    def _init_generator(self):
        cnum = self.config.g_cnum
        # encoder
        encC = [3,cnum,2*cnum,2*cnum,4*cnum,4*cnum,4*cnum]
        encF = [5,4,3,4,3,3]; encS = [1,2,1,2,1,1]
        encP = [2,1,1,1,1,1]
        self.Ge = nn.ModuleList([nn.Conv2d(encC[i],encC[i+1],
            encF[i],encS[i],encP[i]) for i in range(len(encC)-1)])

        # bottleneck
        self.Gb = nn.ModuleList([nn.Conv2d(cnum*4,cnum*4,
                3,1,2**i,2**i) for i in range(1,5)])

        # decoder
        decC = encC[::-1]
        self.Gd = nn.ModuleList([nn.Conv2d(decC[i],decC[i+1],
            3,1,1) for i in range(len(decC)-2)])

        # subpixel_conv
        self.subpixel = nn.Conv2d(cnum,4*cnum,3,1,1,1)        

        # 2
        # encoder
        encC = [68,cnum,cnum,2*cnum,2*cnum,4*cnum,4*cnum]
        encF = [5,4,3,4,3,3]; encS = [1,2,1,2,1,1]
        encP = [2,1,1,1,1,1]
        self.Ge2 = nn.ModuleList([nn.Conv2d(encC[i],encC[i+1],
            encF[i],encS[i],encP[i]) for i in range(len(encC)-1)])

        # bottleneck
        self.Gb2 = nn.ModuleList([nn.Conv2d(cnum*4,cnum*4,
                3,1,2**i,2**i) for i in range(1,5)])

        # decoder
        decC = [4*cnum]+encC[::-1]
        decC[-2] = cnum//2; decC[-1] = 3
        self.Gd2 = nn.ModuleList([nn.Conv2d(decC[i],decC[i+1],
            3,1,1) for i in range(len(decC)-1)])

    def FEN(self,x):
        # input to FEN (x = batch_incomplete)
        # 2,3,128,128 > 2,256,32,32
        x_ = x
        for idx,layer in enumerate(self.Ge):
            x_ = F.elu(layer(x_))
        
        # 2,256,32,32 keep dims
        for idx,layer in enumerate(self.Gb):
            x_ = F.elu(layer(x_))

        # 2,256,32,32 > 2,64,128,128 
        for idx,layer in enumerate(self.Gd):
            if idx in [2,4]:
                x_ = F.interpolate(x_,
                      [x_.shape[2:][0]*2,x_.shape[2:][1]*2])
            x_ = F.elu(layer(x_))
            if idx == 4: break
        return x_

    # padding same, activation nn.elu, nontrainable, no reuse
    def subpixel_conv(self,x,cnum,ksize,target_size,stride=1,rate=1):
        th,tw = target_size
        b,c,h,w = x.shape
        x = self.subpixel(x)
        b,c,h,w = x.shape
        x = x.view(b,c//(th//h*tw//w),th,tw)
        return x
    
    def estimate_meanvar(self,x,mask_,eps):
        x_cnt = torch.max(torch.FloatTensor([eps]).cuda(),
                    torch.sum(mask_,[0,2,3],keepdim=True))
        x_mean = torch.sum(x*mask_,[0,2,3],keepdim=True)/x_cnt
        x_variance = torch.sum((x*mask_-x_mean)**2,[0,2,3],
                    keepdim=True)/x_cnt
        x_mean = Variable(x_mean,requires_grad=False)
        x_variance = Variable(x_variance,requires_grad=False)
        return x_mean.squeeze(),x_variance.squeeze()    

    def context_normalization(self,x,mask,alpha=.5,eps=1e-5):
        mask_s=F.interpolate(1-mask[:,:1,:,:],x.shape[2:])
        x_known_mean,x_known_variance=self.estimate_meanvar(x,mask_s,eps)
        mask_s_rev = 1-mask_s
        x_unknown_mean,x_unknown_variance=self.estimate_meanvar(x,mask_s_rev,eps)
        x_unknown=alpha*F.batch_norm(x*mask_s_rev,x_unknown_mean,x_unknown_variance,torch.sqrt(x_known_variance),x_known_mean,eps=eps,training=False)+(1-alpha)*x*mask_s_rev
        x = x_unknown*mask_s_rev + x*mask_s
        return x
   
    def CPN(self,x_fe,x_in,mask,cnum,use_cn=True,alpha=.5):    
        ones_x = torch.ones_like(x_in)[:,:1,:,:].cuda()
        # 2,68,256,256
        xnow = torch.cat((x_fe,x_in,mask*ones_x),1)
        x_ = xnow
        # 2,68,256,256 > 2,256,64,64
        for idx,layer in enumerate(self.Ge2):
            x_ = F.elu(layer(x_))
        # 2,256,64,64 keep dims
        for idx,layer in enumerate(self.Gb2):
            x_ = F.elu(layer(x_))

        # 2,256,64,64  > 2,3,256,256
        for idx,layer in enumerate(self.Gd2):
            if idx == 1:
                x_ = self.context_normalization(x_,mask)
            if idx in [2,4]:
                x_ = F.interpolate(x_,
                      [x_.shape[2:][0]*2,x_.shape[2:][1]*2])
            x_ = F.elu(layer(x_))
        x_ = torch.clamp(x_,-1,1)
        return x_

    def forward(self,x,mask,margin,config=None,reuse=False):
        cnum = self.config.g_cnum
        use_cn = self.config.use_cn
        fa_alpha = self.config.fa_alpha
        feature_expansion_op = self.subpixel_conv
        target_shape = mask.shape[2:]
        xin_expanded = F.pad(x,(margin.left,margin.right,margin.top,margin.bottom))
        xin_expanded = xin_expanded.view(-1,3,target_shape[0],target_shape[1])
        expand_scale_ratio=int(np.prod(mask.shape[2:])/np.prod(x.shape[2:]))
        
        x_ = self.FEN(x)
        x_fe = feature_expansion_op(x_,cnum*expand_scale_ratio,3,target_shape)
        x_ = self.CPN(x_fe,xin_expanded,mask,cnum,use_cn,fa_alpha)
        return x_,x_fe

class build_contextual_wgan_discriminator(nn.Module):
    def __init__(self,config):
        super(build_contextual_wgan_discriminator,self).__init__()
        self.config = config
        self._init_discriminator()

    def _init_discriminator(self):
        cnum = self.config.d_cnum
        # global
        gloC = [3,cnum,cnum*2,cnum*4,cnum*2]
        self.Dg = nn.ModuleList([nn.Conv2d(gloC[i],gloC[i+1],
            5,2,i%2+1) for i in range(len(gloC)-1)])
        self.Dlin = nn.Linear(32768,1)
        
        # contextual
        conC = [3,cnum,cnum*2,cnum*4]
        self.Dc = nn.ModuleList([nn.Conv2d(conC[i],conC[i+1],
            5,2,2) for i in range(len(conC)-1)])
        self.Dc.append(nn.Conv2d(conC[-1],1,3,1,1))

    def build_wgan_global_discriminator(self,x):
        x_ = x
        for idx,layer in enumerate(self.Dg):
            x_ = F.leaky_relu(layer(x_))
        dglobal = x_.flatten(start_dim=1)
        dout_global = self.Dlin(dglobal)
        return dout_global

    def max_downsampling(self,x,ratio=2):
        iters = math.log2(ratio)
        for _ in range(int(iters)):
            x = F.max_pool2d(x,2)
        return x

    def build_wgan_contextual_discriminator(self,x,mask):
        h,w = x.shape[2:]
        x_ = x
        for idx,layer in enumerate(self.Dc):
            x_ = layer(x_)
            if idx < len(self.Dc)-1:
                x_ = F.leaky_relu(x_)
        mask_ = self.max_downsampling(mask,ratio=8)
        x_ = x_*mask_
        # 2x?,
        x_ = torch.sum(x_,[1,2,3])/torch.sum(mask_,[1,2,3])
        # 1,1,256,256
        mask_local = F.interpolate(mask_,[h,w])
        
        return x_,mask_local

    def forward(self,batch_global,mask):
        dout_global = self.build_wgan_global_discriminator(batch_global)
        dout_local,mask_local = self.build_wgan_contextual_discriminator(batch_global,mask)

        return dout_local,dout_global,mask_local

class SemanticRegenerationNet(nn.Module):
    def __init__(self,config):
        super(SemanticRegenerationNet,self).__init__()
        self.config = config
        self.build_generator = build_generator(config)
        self.build_contextual_wgan_discriminator = build_contextual_wgan_discriminator(config)
        #self.mrfloss = IDMRFLoss()

        self.bbox_gen = random_square
        self.subpixelconv = nn.Conv2d(1,1,64,1,31,bias=False)
        kernel = make_gauss_var(64,1/40).cuda()
        self.subpixelconv.weight = nn.Parameter(kernel,requires_grad=False)

    def relative_spatial_variant_mask(self,mask,hsize=64,sigma=1/40,iters=9):
        eps = 1e-5
        init = 1-mask.clone().detach()

        for i in range(iters):
            #mask_priority = F.pad(self.subpixelconv(init),(0,1,0,1))
            mask_priority = F.interpolate(self.subpixelconv(init),mask.shape[2:])
            mask_priority *= mask
            if i == iters-2:
                mask_priority_pre = mask_priority
            init = mask_priority + (1-mask)
        mask_priority = mask_priority_pre/(mask_priority+eps)
        
        #plt.imshow(compressTensor(mask_priority)); plt.show()
        return mask_priority 

    def random_interpolates(self,x,y,alpha=None):
        shape = x.shape
        x = x.view(shape[0],-1)
        y = y.view(shape[0],-1)
        if alpha is None:
            alpha = torch.rand((shape[0],1)).cuda()
        interpolates = x+alpha*(y-x)
        interpolates = interpolates.view(shape)
        return interpolates

    def gan_wgan_loss(self,pos,neg):
        d_loss = torch.mean(neg-pos)
        g_loss = -torch.mean(neg)
        return g_loss,d_loss
    
    def gradients_penalty(self,x,y,mask=None,norm=1):
        #gradients = torch.autograd.grad(y[0],x)[0] # need to check
        gradients = torch.autograd.grad(y,x,create_graph=True,grad_outputs=torch.ones(y.size()).cuda(),retain_graph=True,only_inputs=True)[0]
        if mask is None:
            mask = torch.ones_like(gradients)
        slopes=torch.sqrt(torch.sum(torch.square(gradients)*mask,[1,2,3]))
        return torch.mean(torch.square(slopes-norm))

    def updateMask(self,mask,mask_priority,margin):
        self.mask,self.margin = mask,margin
        self.mask_priority = mask_priority

        self.build_generator.mask = mask
        self.build_contextual_wgan_discriminator.mask = mask
        self.build_generator.mask_priority = mask_priority
        self.build_contextual_wgan_discriminator.mask_priority = mask_priority

    def forwardD(self,x,batch_complete,mask,losses):
        # gan
        batch_pos_neg = torch.cat((x,batch_complete),0)

        # wgan with gradient penalty
        build_critics = self.build_contextual_wgan_discriminator
        # separate gan
        global_wgan_loss_alpha = 1.0
        pos_neg_local,pos_neg_global,mask_local = build_critics(batch_pos_neg,mask)
        try:
            pos_local,neg_local = torch.split(pos_neg_local,2)
            pos_global,neg_global = torch.split(pos_neg_global,2)
        except:
            pos_local,neg_local = torch.split(pos_neg_local,2)[0]
            pos_global,neg_global = torch.split(pos_neg_global,2)[0]

        # gp ?,3,256,256
        interpolates_global = self.random_interpolates(x,batch_complete)
        interpolates_local = interpolates_global
        dout_local,dout_global,_ = build_critics(interpolates_global,mask)

        # apply penalty
        penalty_local = self.gradients_penalty(interpolates_local,dout_local,mask=mask_local)
        penalty_global = self.gradients_penalty(interpolates_global,dout_global,mask=mask)

        # loss calculation for wgan discriminator
        g_loss_local,d_loss_local = self.gan_wgan_loss(pos_local,neg_local)
        g_loss_global,d_loss_global = self.gan_wgan_loss(pos_global,neg_global)
        losses['d_loss'] = d_loss_global+d_loss_local
        losses['gp_loss'] = self.config.wgan_gp_lambda*(penalty_local+penalty_global)
        losses['d_loss'] += losses['gp_loss']
        return g_loss_local,d_loss_local,g_loss_global,g_loss_global,losses

    def forwardG(self,x,batch_incomplete,mask,mask_priority,margin,losses):
        # generator
        x_,x_fe = self.build_generator(batch_incomplete,mask,margin)
        batch_predicted = x_ 
        batch_complete = batch_predicted*mask + x*(1-mask)
        
        # no pretrain
        self.config.feat_style_layers = {'conv3_2':1.0,'conv4_2':1.0}
        self.config.feat_content_layers = {'conv4_2':1.0}
        self.config.mrf_style_w = 1.0
        self.config.mrf_content_w = 1.0
            
        #losses['id_mrf_loss'] = self.mrfloss(batch_predicted,x)

        # loss calculation for generator
        losses['l1_loss'] = self.config.pretrain_l1_alpha*torch.mean(torch.abs(x-x_)*mask_priority)
        losses['ae_loss'] = self.config.pretrain_l1_alpha*torch.mean(torch.abs(x-x_)*(1-mask))
        losses['ae_loss'] /= torch.mean(1-mask)

        g_loss_local,d_loss_local,g_loss_global,g_loss_global,losses = self.forwardD(x,batch_complete,mask,losses)

        # visualization
        global_wgan_loss_alpha = 1
        batch_incomplete_pad = F.pad(batch_incomplete,(margin.left,margin.right,margin.top,margin.bottom))
        viz_img = torch.cat([x[0],batch_incomplete_pad[0],batch_complete[0]],axis=2)
        viz_img = compressTensor(viz_img).transpose(1,2,0)*127.5+127.5
        losses['g_loss'] = global_wgan_loss_alpha*g_loss_global+g_loss_local
        losses['g_loss'] *= self.config.gan_loss_alpha
        #losses['g_loss'] += self.config.mrf_alpha*losses['id_mrf_loss']
        losses['g_loss'] += self.config.l1_loss_alpha*losses['l1_loss']
        losses['g_loss'] += self.config.ae_loss_alpha*losses['ae_loss']
        return losses,viz_img

    def forward(self,x,oG=None,oD=None):
        # mask for cropping 
        bbox,margin = self.bbox_gen(self.config)
        mask = bbox2mask(bbox,args)
        mask = torch.tensor(1 - mask,requires_grad=False).cuda()
        h,w = x.shape[2:]
        batch_incomplete=x[:,:,margin.top:margin.top+self.config.mask_shapes[0],margin.left:margin.left+self.config.mask_shapes[1]]

        mask_priority = self.relative_spatial_variant_mask(mask)
        self.updateMask(mask,mask_priority,margin)
        x_,x_fe = self.build_generator(batch_incomplete,mask,margin)
        batch_predicted = x_ 

        losses = {}
        batch_complete = batch_predicted*mask + x*(1-mask)
        # ----------------------------------------------------------
        if oD is not None and oG is not None:
            for i in range(5):
                oD.zero_grad();oG.zero_grad()
                _,_,_,_,losses = self.forwardD(x,batch_complete,mask,losses)
                losses['d_loss'].backward(retain_graph=True)
                oD.step()
        
            oG.zero_grad()
            losses,viz_img=self.forwardG(x,batch_incomplete,mask,mask_priority,margin,losses)
            losses['g_loss'].backward()
            oG.step()
        else:
            _,_,_,_,losses = self.forwardD(x,batch_complete,mask,losses)
            losses,viz_img=self.forwardG(x,batch_incomplete,mask,mask_priority,margin,losses)
        return losses,viz_img

if __name__ == "__main__":
    #with torch.no_grad():
    testin = Variable(torch.randn(2,3,256,256),requires_grad=True).cuda()
    srnet = SemanticRegenerationNet(args).cuda()
    #srnet.train()
    losses,viz_img = srnet.forward(testin)
    pdb.set_trace()


