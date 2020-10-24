import argparse
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log/store')
import os
from net import *
from data import *
import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

def generateViz(x):
    f = plt.figure()
    plt.imshow(x);plt.axis('off')
    return f

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data',type=str,default='/media/yui/Disk/data/cat2dog/')
    p.add_argument('--epoch',type=int,default=1000)
    p.add_argument('--mode',type=str,default='train',choices=['train','test'])
    p.add_argument('--img_shapes',nargs='+',default=[256,256,3])
    p.add_argument('--mask_shapes',nargs='+',default=[128,128])
    p.add_argument('--batch_size',type=int,default=4)
    p.add_argument('--feat_expansion_op',type=str,default='subpixel')
    p.add_argument('--use_cn',type=int,default=1)
    p.add_argument('--g_cnum',type=int,default=64)
    p.add_argument('--d_cnum',type=int,default=64)
    p.add_argument('--gan_loss_alpha',type=float,default=1e-3)
    p.add_argument('--wgan_gp_lambda',type=float,default=10)
    p.add_argument('--pretrain_l1_alpha',type=float,default=1.2)
    p.add_argument('--l1_loss_alpha',type=float,default=4.2)
    p.add_argument('--ae_loss_alpha',type=float,default=1.2)
    p.add_argument('--mrf_alpha',type=float,default=0.05)
    p.add_argument('--fa_alpha',type=float,default=0.5)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--l1_type',type=int,default=0)
    p.add_argument('--random_mask',type=int,default=1)
    p.add_argument('--max_delta_shapes',nargs='+',default=[0,0])
    p.add_argument('--margins',nargs='+',default=[0,0])
    args = p.parse_args()

    if args.mode == "train":
        # Setup
        trans = transforms.Compose([
                    transforms.Resize((256,256)),transforms.ToTensor()])
        dog = dogDataset(transform=trans)
        dataloader = DataLoader(dog,batch_size=args.batch_size,shuffle=True)
        model = SemanticRegenerationNet(args).cuda()
        optimG = optim.Adam(model.build_generator.parameters(),lr=1e-5,betas=(.5,.9))
        optimD = optim.Adam(model.build_contextual_wgan_discriminator.parameters(),lr=5e-5,betas=(.5,.9))

        # Training loop
        ite = 0
        for epoch in range(args.epoch):
            for idx,im in enumerate(tqdm.tqdm(dataloader)):
                im = im.cuda()
                losses,viz = model(im,optimG,optimD)
            # tensorboard
                if idx%5 == 0:
                    if idx%20 == 0:
                        f = generateViz((viz*255).astype(np.uint8))
                        writer.add_figure('train {}'.format(ite),f)
                    writer.add_scalars('loss',{'g_loss':losses['g_loss'].item(),
                        'd_loss':losses['d_loss'].item()},ite)
                ite+=1
            # save parameters
            if epoch%20 == 0:
                torch.save(model.build_generator.state_dict(),'log/store/G.pt')
                torch.save(model.build_contextual_wgan_discriminator.state_dict(),'log/store/D.pt')
            #pdb.set_trace()           
    elif args.mode == "test":
        pass




