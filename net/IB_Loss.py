import torch
import torch.nn as nn
import torch.nn.functional as F

class IBLoss(nn.Module):
    def __init__(self, temperature=1, reduction='mean'):
        super(IBLoss, self).__init__()
        self.temperature =temperature
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.kld = nn.KLDivLoss(reduction=reduction)
        self.CE = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, z, v, mask):
        b, c, _, _ = z.size()
        loss_CE = self.CE(z, mask)
        z = z.reshape(b * c, -1)
        v = v.reshape(b * c, -1)
        loss_kld = self.kld(self.logsoftmax(v.detach() / self.temperature),
                            self.softmax(z / self.temperature))
        loss_IB = loss_kld + loss_CE
        
        return loss_IB
    
if __name__ == '__main__':
    IB_Loss = IBLoss(temperature=1, reduction='mean').cuda()
    z = torch.randn(16, 1, 384, 384).cuda()  
    v = torch.randn(16, 1, 384, 384).cuda()  
    mask = torch.randn(16, 1, 384, 384).cuda()
    Loss = IB_Loss(z, v, mask)
    print(Loss)