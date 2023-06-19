import torch 
import torch.nn as nn
import torch.nn.functional as F
import json

import glob
import os
import nibabel as nib


import numpy as np
import struct
import csv

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


def get_validations(json_file):
    with open(json_file) as f:
        dataset=json.load(f)
    out=[]
    for x in dataset['registration_val']:
        out.append(x['fixed'])
        out.append(x['moving'])
    return sorted(list(set(out)))

def get_dataset_split(dataset):
    validation=[]; training=[]
    for x in dataset['registration_val']:
        validation.append(x['fixed'])
        validation.append(x['moving'])
    validation=sorted(list(set(validation)))
    training = [x['image'] for x in dataset['training']]
    training=sorted(list(set(training)-set(validation)))
    return training,validation

def load_data(task, data_path='data_compressed'):
    if task =='AbdomenCTCT' or task == 'AbdomenMRMR':
        ##dataset_info
        with open(f'{data_path}/{task}/{task}_dataset.json') as f:
            dataset=json.load(f)
        training,validation=get_dataset_split(dataset)
        image_shape=dataset['tensorImageShape']['0']
        num_labels=len(dataset['labels']['0'])
        H,W,D=image_shape

            # allocate data
        tmp_gt_label_tr = torch.zeros(len(training), num_labels, H//2, W//2, D//2).pin_memory()
        tmp_gt_label_val = torch.zeros(len(validation), num_labels, H, W, D).pin_memory()
        tmp_pred_label_val = torch.zeros(len(validation), num_labels, H//2, W//2, D//2).pin_memory()

        for ii, i in enumerate(tqdm(training)):
            tmp = torch.from_numpy(nib.load(os.path.join(data_path,task,'labelsTr',os.path.basename(i))).get_fdata()).float().cuda()
            onehot = F.one_hot(tmp.long(),num_labels).permute(3,0,1,2).unsqueeze(0).half()
            tmp_gt_label_tr[ii] = F.avg_pool3d(onehot,2,stride=2).cpu().float()


        for ii, i in enumerate(tqdm(validation)):

            tmp2 = torch.from_numpy(nib.load(os.path.join(data_path,task,'labelsTr',os.path.basename(i))).get_fdata()).float().cpu()
            onehot = F.one_hot(tmp2.long(),num_labels).permute(3,0,1,2).unsqueeze(0)
            tmp_gt_label_val[ii] =onehot.cpu().float()

            tmp2 = torch.from_numpy(nib.load(os.path.join(data_path,task,'predictedlabelsTr',os.path.basename(i))).get_fdata()).float().cuda()
            onehot = F.one_hot(tmp2.long(),num_labels).permute(3,0,1,2).unsqueeze(0).half()
            tmp_pred_label_val[ii] = F.avg_pool3d(onehot,2,stride=2).cpu().float()

        gt_label_tr = tmp_gt_label_tr
        gt_label_val = tmp_gt_label_val
        pred_label_val = tmp_pred_label_val

    if task == 'TS_Skeletal':

        list_data=sorted(glob.glob(f'{data_path}/TS_Skeletal/labels/*nii.gz'))

        training = list_data[:27]
        validation = list_data[27:]
        num_labels ,H,W,D = (29,256, 160, 256)
        tmp_gt_label_tr = torch.zeros(len(training), num_labels, H//2, W//2, D//2)#.pin_memory()
        tmp_gt_label_val = torch.zeros(len(validation), num_labels, H, W, D)#.pin_memory()
        tmp_pred_label_val = torch.zeros(len(validation), num_labels, H//2, W//2, D//2)#.pin_memory()
        
        for ii, i in enumerate(tqdm(training)):
            tmp = torch.from_numpy(nib.load(i).get_fdata()).float().cuda()

            onehot = F.one_hot(tmp.long(),num_labels).permute(3,0,1,2).unsqueeze(0).half()
            tmp_gt_label_tr[ii] = F.avg_pool3d(onehot,2,stride=2).cpu().float()

        for ii, i in enumerate(tqdm(validation)):

            tmp = torch.from_numpy(nib.load(i).get_fdata()).float().cuda()
            onehot = F.one_hot(tmp.long(),num_labels).permute(3,0,1,2).unsqueeze(0)
            tmp_gt_label_val[ii] =onehot.cpu().float()

            tmp = torch.from_numpy(nib.load(i).get_fdata()).float().cuda()
            onehot = F.one_hot(tmp.long(),num_labels).permute(3,0,1,2).unsqueeze(0).half()
            tmp_pred_label_val[ii] = F.avg_pool3d(onehot,2,stride=2).cpu().float()
        
        gt_label_tr = tmp_gt_label_tr
        gt_label_val = tmp_gt_label_val
        pred_label_val = tmp_pred_label_val

    return gt_label_tr, gt_label_val, pred_label_val

def soft_dice(fixed,moving, omit_bg=False):
    B,C,H,W,D = fixed.shape
    dice = (2. * fixed[:,int(omit_bg):]*moving[:,int(omit_bg):]).reshape(B,-1,H*W*D).mean(2) / (1e-8 + fixed[:,int(omit_bg):].reshape(B,-1,H*W*D).mean(2) + moving[:,int(omit_bg):].reshape(B,-1,H*W*D).mean(2))
    return dice


def AdamRegMIND(mind_fix,mind_mov,dense_flow,mask_fix=1,grid_sp = 2,data_weight=1):
    
    if(dense_flow.shape[-1]==3):
        dense_flow = dense_flow.permute(0,4,1,2,3)
    
    H,W,D = dense_flow[0,0].shape
    
    disp_hr = dense_flow.cuda().flip(1)*torch.tensor([H-1,W-1,D-1]).cuda().view(1,3,1,1,1)/2
    with torch.enable_grad(): 
        disp_lr = F.interpolate(disp_hr,size=(H//grid_sp,W//grid_sp,D//grid_sp),mode='trilinear',align_corners=False)
        net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp,W//grid_sp,D//grid_sp),bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1)
        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp,W//grid_sp,D//grid_sp),align_corners=False)
        #run Adam optimisation with diffusion regularisation and B-spline smoothing
        lambda_weight = .65#.65
        #patch_fix_sampled = F.grid_sample(mind_fix.cuda().float(),grid0.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda()\
        #                                      ,align_corners=False,mode='bilinear')
        num_iter = 70
        #num_iter = 50 #submission
        ramp_up = torch.sigmoid(torch.linspace(-15,7,num_iter))
        for iter in range(num_iter):
            optimizer.zero_grad()
            disp_sample = (mask_fix*F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),5,stride=1,padding=2),\
                                       5,stride=1,padding=2)).permute(0,2,3,4,1)
            #reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
            #lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
            #lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()
            reg_loss = ramp_up[iter]*1e-4*jacobian_determinant_3d(disp_sample.permute(0,4,1,2,3)).std()
            scale = torch.tensor([(H//grid_sp-1)/2,(W//grid_sp-1)/2,(D//grid_sp-1)/2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()
            patch_mov_sampled = F.grid_sample(mind_mov.cuda().float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda()\
                                              ,align_corners=False,mode='bilinear')
            #sampled_cost = (patch_mov_sampled-patch_fix_sampled.cuda()).pow(2).mean(1)*12
            sampled_cost = (patch_mov_sampled-mind_fix.cuda().float()).pow(2).mean(1)*data_weight
            loss = sampled_cost.mean()
            (loss+reg_loss).backward()
            optimizer.step()
        fitted_grid = disp_sample.permute(0,4,1,2,3).detach()
        disp_hr = F.interpolate(fitted_grid*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)

    disp_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,3,padding=1,stride=1),3,padding=1,stride=1),3,padding=1,stride=1)


    disp_hr = torch.flip(disp_smooth/torch.tensor([H-1,W-1,D-1]).view(1,3,1,1,1).cuda()*2,[1])
    return disp_hr

def jacobian_determinant_3d(dense_flow):
    B,_,H,W,D = dense_flow.size()
    
    dense_pix = dense_flow*(torch.Tensor([H-1,W-1,D-1])/2).view(1,3,1,1,1).to(dense_flow.device)
    gradz = nn.Conv3d(3,3,(3,1,1),padding=(1,0,0),bias=False,groups=3)
    gradz.weight.data[:,0,:,0,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    gradz.to(dense_flow.device)
    grady = nn.Conv3d(3,3,(1,3,1),padding=(0,1,0),bias=False,groups=3)
    grady.weight.data[:,0,0,:,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    grady.to(dense_flow.device)
    gradx = nn.Conv3d(3,3,(1,1,3),padding=(0,0,1),bias=False,groups=3)
    gradx.weight.data[:,0,0,0,:] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    gradx.to(dense_flow.device)
    
    with torch.no_grad():
        jacobian = torch.cat((gradz(dense_pix),grady(dense_pix),gradx(dense_pix)),0)+torch.eye(3,3).view(3,3,1,1,1).to(dense_flow.device)
        jacobian = jacobian[:,:,2:-2,2:-2,2:-2]
        jac_det = jacobian[0,0,:,:,:]*(jacobian[1,1,:,:,:]*jacobian[2,2,:,:,:]-jacobian[1,2,:,:,:]*jacobian[2,1,:,:,:])-jacobian[1,0,:,:,:]*(jacobian[0,1,:,:,:]*jacobian[2,2,:,:,:]-jacobian[0,2,:,:,:]*jacobian[2,1,:,:,:])+jacobian[2,0,:,:,:]*(jacobian[0,1,:,:,:]*jacobian[1,2,:,:,:]-jacobian[0,2,:,:,:]*jacobian[1,1,:,:,:])

    return jac_det


def load_dataset(task_name, split='train'):
    if task_name == 'NLST':
        spacing=torch.tensor([1.5,1.5,1.5])
        H=D=224; W=192
        if split == 'train':
            cases = [str(x).zfill(4) for x in range(1,101)]
            mode = 'Tr'
        elif split == 'val':
            cases = [str(x).zfill(4) for x in range(101,111)]
            mode = 'Tr'
        elif split == 'all':
            cases = [str(x).zfill(4) for x in range(1,111)]
            mode = 'Tr'
        elif split == 'test':
            cases = [str(x).zfill(4) for x in range(111,141)]
            mode = 'Ts'

        num_train=len(cases)
        label_fix = torch.zeros(num_train,2,H//2,W//2,D//2)
        label_mov = torch.zeros(num_train,2,H//2,W//2,D//2)

        kpts_fix = []
        kpts_mov = []

        for i,case in tqdm(enumerate(cases),total=num_train):
            label_fix[i] = F.avg_pool3d(F.one_hot(torch.from_numpy(nib.load(f'data_compressed/NLST/labels{mode}/NLST_{case}_0000/lung_vessels.nii.gz').get_fdata()).long().unsqueeze(0),2).permute(0,4,1,2,3),2)
            label_mov[i] = F.avg_pool3d(F.one_hot(torch.from_numpy(nib.load(f'data_compressed/NLST/labels{mode}/NLST_{case}_0001/lung_vessels.nii.gz').get_fdata()).long().unsqueeze(0),2).permute(0,4,1,2,3),2)

            kpts_fix_,kpts_mov_,_,_ = read_cf2(f'data_compressed/NLST/keypoints{mode}/NLST_{case}',H,W,D)
            kpts_fix.append(kpts_fix_)
            kpts_mov.append(kpts_mov_)


    elif task_name == 'Lung_CT':
        spacing=torch.tensor([1.25,1.25,1.25])
        H=W=192; D=208
        if split == 'train':
            cases = [str(x).zfill(4) for x in range(4,21)]
            mode = 'Tr' ; grid_sp = 2
        elif split == 'val':
            cases = [str(x).zfill(4) for x in range(1,4)]
            mode = 'Tr' ; grid_sp = 2
        elif split == 'all':
            cases = [str(x).zfill(4) for x in range(1,21)]
            mode = 'Tr' ; grid_sp = 2
        elif split == 'test':
            cases = [str(x).zfill(4) for x in range(21,31)]
            mode = 'Ts' ; grid_sp = 2

        
        num_train=len(cases)
        label_fix = torch.zeros(num_train,2,H//grid_sp,W//grid_sp,D//grid_sp)
        label_mov = torch.zeros(num_train,2,H//grid_sp,W//grid_sp,D//grid_sp)

        kpts_fix = []
        kpts_mov = []

        for i,case in tqdm(enumerate(cases),total=num_train):
            if grid_sp == 2:
                label_fix[i] = F.avg_pool3d(F.one_hot(torch.from_numpy(nib.load(f'data_compressed/LungCT/labels{mode}/LungCT_{case}_0000/lung_vessels.nii.gz').get_fdata()).flip(0,1).long().unsqueeze(0),2).permute(0,4,1,2,3),2)
                label_mov[i] = F.avg_pool3d(F.one_hot(torch.from_numpy(nib.load(f'data_compressed/LungCT/labels{mode}/LungCT_{case}_0001/lung_vessels.nii.gz').get_fdata()).flip(0,1).long().unsqueeze(0),2).permute(0,4,1,2,3),2)
            elif grid_sp == 1:
                label_fix[i] = F.one_hot(torch.from_numpy(nib.load(f'data_compressed/LungCT/labels{mode}/LungCT_{case}_0000/lung_vessels.nii.gz').get_fdata()).flip(0,1).long().unsqueeze(0),2).permute(0,4,1,2,3)
                label_mov[i] = F.one_hot(torch.from_numpy(nib.load(f'data_compressed/LungCT/labels{mode}/LungCT_{case}_0001/lung_vessels.nii.gz').get_fdata()).flip(0,1).long().unsqueeze(0),2).permute(0,4,1,2,3)
            

            kpts_fix_,kpts_mov_,_,_ = read_cf2(f'data_compressed/LungCT/keypoints{mode}/LungCT_{case}',H,W,D)
            kpts_fix.append(kpts_fix_*torch.tensor([1,-1,-1]))
            kpts_mov.append(kpts_mov_*torch.tensor([1,-1,-1]))


    elif task_name == 'COPDgene':
        spacing = torch.tensor([1.75,1.25,1.75])
        H=W=192; D=208
        if split == 'all':
            cases = [str(x).zfill(2) for x in range(1,11)]
        elif split == 'train':
            cases = [str(x).zfill(2) for x in range(1,9)]
        elif split == 'val':
            cases = [str(x).zfill(2) for x in range(9,11)]
        else:
            print('no split')
            pass
        num_train=len(cases)
        label_fix = torch.zeros(num_train,2,H//2,W//2,D//2)
        label_mov = torch.zeros(num_train,2,H//2,W//2,D//2)


        kpts_fix = []
        kpts_mov = []
        for i,case in tqdm(enumerate(cases),total=num_train):
            label_fix[i] = F.avg_pool3d(F.one_hot(torch.from_numpy(nib.load(f'data_compressed/COPDgene/vessels/case_{case}_exp.nii.gz').get_fdata()).long().unsqueeze(0),2).permute(0,4,1,2,3),2)
            label_mov[i] = F.avg_pool3d(F.one_hot(torch.from_numpy(nib.load(f'data_compressed/COPDgene/vessels/case_{case}_insp.nii.gz').get_fdata()).long().unsqueeze(0),2).permute(0,4,1,2,3),2)


            with open(f'data_compressed/COPDgene/keypoints/case_{case}_insp.dat', 'rb') as content_file:
                content = content_file.read()
            corrfield = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(-1,6).float()
            kpts_mov_ = torch.stack((corrfield[:,2+0]/(D-1)*2-1,corrfield[:,1+0]/(W-1)*2-1,corrfield[:,0+0]/(H-1)*2-1),1)
            kpts_fix_ = torch.stack((corrfield[:,2+3]/(D-1)*2-1,corrfield[:,1+3]/(W-1)*2-1,corrfield[:,0+3]/(H-1)*2-1),1)

            kpts_fix.append(kpts_fix_)
            kpts_mov.append(kpts_mov_)


    elif task_name == 'EMPIRE10':
        spacing = torch.tensor([1.75,1.25,1.75])
        H=W=192; D=208
        if split == 'all':
            cases = [str(x).zfill(2) for x in [1,7,8,14,18,20,21]]
        elif split == 'train':
            cases = [str(x).zfill(2) for x in [1,7,8,14,18]]
        elif split == 'val':
            cases = [str(x).zfill(2) for x in [20,21]]
        else:
            print('no split')
            pass
        num_train=len(cases)
        label_fix = torch.zeros(num_train,2,H//2,W//2,D//2)
        label_mov = torch.zeros(num_train,2,H//2,W//2,D//2)


        kpts_fix = []
        kpts_mov = []
        for i,case in tqdm(enumerate(cases),total=num_train):
            label_fix[i] = F.avg_pool3d(F.one_hot(torch.from_numpy(nib.load(f'data_compressed/EMPIRE10/vessels/case_{case}_exp.nii.gz').get_fdata()).long().unsqueeze(0),2).permute(0,4,1,2,3),2)
            label_mov[i] = F.avg_pool3d(F.one_hot(torch.from_numpy(nib.load(f'data_compressed/EMPIRE10/vessels/case_{case}_insp.nii.gz').get_fdata()).long().unsqueeze(0),2).permute(0,4,1,2,3),2)


            with open(f'data_compressed/EMPIRE10/keypoints/case_{case}.dat', 'rb') as content_file:
                content = content_file.read()
            corrfield = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(-1,6).float()
            kpts_fix_ = torch.stack((corrfield[:,2+0]/207*2-1,corrfield[:,1+0]/191*2-1,corrfield[:,0+0]/191*2-1),1)
            kpts_mov_ = torch.stack((corrfield[:,2+3]/207*2-1,corrfield[:,1+3]/191*2-1,corrfield[:,0+3]/191*2-1),1)
            #kpts_mov_ = torch.stack((corrfield[:,2+0]/(D-1)*2-1,corrfield[:,1+0]/(W-1)*2-1,corrfield[:,0+0]/(H-1)*2-1),1).cuda()
            #kpts_fox_ = torch.stack((corrfield[:,2+3]/(D-1)*2-1,corrfield[:,1+3]/(W-1)*2-1,corrfield[:,0+3]/(H-1)*2-1),1).cuda()

            kpts_fix.append(kpts_fix_)
            kpts_mov.append(kpts_mov_)

    print(task_name,split)
    return label_fix,label_mov,kpts_fix,kpts_mov,(num_train,H,W,D),spacing


def read_cf2(filename,H,W,D):
    corrfield = torch.empty(0,3)
    with open(filename+'_0000.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i,row in enumerate(csvreader):
            corrfield = torch.cat((corrfield,torch.from_numpy(np.array(row).astype('float32')).view(1,-1)),0)
   
    kpts_fix = ((corrfield[:,:]+.5)/torch.tensor([H/2,W/2,D/2])).flip(-1)-1
    corrfield2 = torch.empty(0,3)
    with open(filename+'_0001.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i,row in enumerate(csvreader):
            corrfield2 = torch.cat((corrfield2,torch.from_numpy(np.array(row).astype('float32')).view(1,-1)),0)
   
    kpts_mov = ((corrfield2[:,:]+.5)/torch.tensor([H/2,W/2,D/2])).flip(-1)-1
    
    return kpts_fix,kpts_mov,corrfield,corrfield2
##to flip images?