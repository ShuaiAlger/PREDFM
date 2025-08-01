
import torch
import torch.nn as nn


def shift_topk_candidate(kpts, desc, B, K, G, topk=1):
    shifts= torch.topk(desc[:,:,0,:], k= topk, dim=2)[1]
    desc = desc.reshape(B*K, -1, G)
    shifts = shifts.reshape(B*K, -1)

    kpts_update = kpts.repeat(1, topk, 1) ## [B, topk*K, 2] 
    desc_update = []
    for shift in shifts.t():
        for d, s in zip(desc, shift):
            desc_update.append(torch.roll(d, shifts=-int(s), dims=-1)) ## reverse shift

    desc_update = torch.stack(desc_update) ## [topk*B*K, C, G]
    desc_update = desc_update.reshape(topk,B,K,-1,G).transpose(0,1) ## [B*topk*K, C, G]   
    desc_update = desc_update.reshape(B, topk*K,-1)
    return kpts_update, desc_update

def shift_ratio_candidate(kpts, desc, B, K, CG, G, ratio=1.0):
    ## Warning: ratio sampling make different number of keypoints in an image.
    assert desc.min() >= 0
    value, _ = torch.max(desc[:, :, 0, :], dim=2)
    ratio_tensor = (desc[:, :, 0, :] / value.unsqueeze(-1))  ## obtain ratio
    ratio_mask = ratio_tensor >= ratio

    ### un-batchfying because the number of keypoints are different.
    kpts_update = []
    desc_update = []
    for _kpts, _desc, _ratio_mask in zip(kpts, desc, ratio_mask):
        kpts_update_iter = []
        desc_update_iter = []
        for k, d, r in zip(_kpts, _desc, _ratio_mask):
            shifts = r.nonzero().reshape(-1)
            for s in shifts:
                kpts_update_iter.append(k)
                desc_update_iter.append(torch.roll(d, shifts=-int(s), dims=-1))
            
        kpts_update.append(torch.stack(kpts_update_iter))
        desc_update.append(torch.stack(desc_update_iter).reshape(-1, CG))
    return kpts_update, desc_update


def shifting(desc, shift, B, K, G):
    desc = desc.reshape(B*K, -1, G)
    shift = shift.reshape(B*K)
    
    desc_update = []
    for d, s in zip(desc, shift):
        desc_update.append(torch.roll(d, shifts=-int(s), dims=-1)) ## reverse shift
    desc_update = torch.stack(desc_update).reshape(B,K,-1)
    return desc_update


class DescGroupPoolandNorm:
    def __init__(self, args):
        self.pool = 'shift'
        self.norm = "l2"
        self.candidate = "top1"

        self.G = args['num_group']


    def desc_pool_and_norm_infer(self, kpts, desc):

        """ kpts [B, K, 2], desc torch.tensor([B, K, CG])
        """
        B, K, CG = desc.shape
        desc = desc.reshape(B,K,-1,self.G)

        if self.pool == 'shift':
            if 'top' in self.candidate:
                topk = int(self.candidate[3])
                kpts_update, desc_update = shift_topk_candidate(kpts, desc, B, K, self.G, topk)
            else:  ## ratio
                ratio = float(self.candidate)
                kpts_update, desc_update = shift_ratio_candidate(kpts, desc, B, K, CG, self.G, ratio)
        else:
            kpts_update = kpts
            desc_update = self._descriptor_pool(desc)

        if 'top' in self.candidate or self.pool!='shift': ## batchified version
            desc_update = self._descriptor_norm(desc_update)
        else: ## ratio (batch as list)
            desc_update_list = []
            for dd in desc_update:
                desc_update_list.append( self._descriptor_norm(dd))
            desc_update = desc_update_list
        
        return kpts_update, desc_update

    def desc_pool_and_norm(self, desc, GTShift):
        """ desc [B, K, CG], for training.
        """

        B, K, CG = desc.shape
        desc = desc.reshape(B,K,-1,self.G)

        if self.pool=='shift':
            shift = GTShift.unsqueeze(1).repeat(1, K)       
            desc_update = shifting(desc, shift, B, K, self.G)
        else:
            desc_update = self._descriptor_pool(desc)

        desc_update = self._descriptor_norm(desc_update)

        self._exception_handling(desc_update, desc, B, K, CG, shift)

        return desc_update

    def _descriptor_pool(self, desc):
        B, K, CG = desc.shape
        if self.pool=='avg':
            desc_update = desc.mean(dim=3)
        elif self.pool=='weight_avg':
            weight = desc[:, :, 0, :].unsqueeze(2) 
            desc_update = (weight * desc).mean(dim=3)
        elif self.pool=='max':
            desc_update = desc.max(dim=3)[0]
        elif self.pool=='norm':
            desc_update = desc.norm(dim=3)
        elif self.pool=='none':
            desc_update = desc.reshape(B, K, CG)
        else:
            raise NotImplementedError
        return desc_update

    def _descriptor_norm(self, desc_update):
        ## descriptor normalize; default: l2 normalize
        if self.norm=='l2':
            desc_update = L2Norm()(desc_update)
        elif self.norm =='l1':
            desc_update = L1Norm()(desc_update)
        elif self.norm =='context':
            desc_update = ContextNorm()(desc_update)
        elif self.norm == 'none':
            desc_update = desc_update
        else:
            raise NotImplementedError
        return desc_update


    def _exception_handling(self, desc_update, desc, B, K, CG, shift):
        if self.pool == 'shift' or self.pool =='shift_avg' or self.pool =='shift_all':
            assert desc_update.shape == desc.reshape(B,K,CG).shape
            assert shift.max() < self.G and shift.min() >=0
        elif  self.pool =='none':
            assert desc_update.shape == desc.reshape(B,K,CG).shape
        elif self.pool == 'bilinear':
            assert desc_update.shape[-1] == 128
        else:
            assert desc_update.shape[:3] == desc.shape[:3]  ## [B, K, C]


######## descriptor normalize

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-6

    def forward(self, x):
        """ x [B, K, CG] or [K, CG]"""
        # norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        norm = torch.norm(x, p=2, dim=-1) + self.eps
        x = x / norm.unsqueeze(-1)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-6

    def forward(self, x):
        # norm = torch.sum(torch.abs(x), dim=1) + self.eps
        norm = torch.norm(x, p=1, dim=-1) + self.eps
        x = x / norm.unsqueeze(-1)
        return x

class ContextNorm(nn.Module):
    def __init__(self):
        super(ContextNorm, self).__init__()
        self.eps = 1e-6
    
    def forward(self, x):
        mean = torch.mean(x, dim=-1) 
        std = torch.std(x, dim=-1) + self.eps
        x = (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        return x