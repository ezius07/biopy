import torch
from torch import nn

class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect'):
        
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

        
class ExprDecHead(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=4, stride=3, padding=0),
            ConvBNReLU(8, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        )
            
        self.pool = nn.MaxPool1d(kernel_size=4, stride=3, padding=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
    

class ExprAEClassifier(nn.Module):
    
    def __init__(self, num_classes):
        
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=4, stride=3, padding=2),
            ConvBNReLU(128, 256, kernel_size=1, stride=2, padding=1),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class JointLatentGenerator(nn.Module):
    
    def __init__(self, num_classes, expr_ae, img_ae, img_ae_backbone, img_ae_trainable_blocks_bkbn=3, pretrained=True,
                 hidden_size=16, mrna_ds=58276, mirna_ds=58276, meth_ds=58276, img_ds=(3, 400, 400)):
        
        super().__init__()
        
        self.mrna_ae  = expr_ae(data_size=mrna_ds,  hidden_size=hidden_size)
        self.mirna_ae = expr_ae(data_size=mirna_ds, hidden_size=hidden_size)
        self.meth_ae  = expr_ae(data_size=meth_ds,  hidden_size=hidden_size)
        self.img_ae   = img_ae(img_ae_backbone, img_ae_trainable_blocks_bkbn, pretrained=pretrained,
                               hidden_size=hidden_size, output_size=img_ds)
        
        self.mrna_head  = ExprDecHead()
        self.mirna_head = ExprDecHead()
        self.meth_head  = ExprDecHead()
        
        self.ae_classifier = ExprAEClassifier(num_classes)
     
    def __img_predict(self, img_latent, omics_type='mrna'):
        
        if omics_type == 'mrna':
            decoder = self.mrna_ae.decoder
            head = self.mrna_head
        elif omics_type == 'mirna':
            decoder = self.mirna_ae.decoder
            head = self.mirna_head
        elif omics_type == 'meth':
            decoder = self.meth_ae.decoder
            head = self.meth_head
        else:
            raise NotImplementedError("Specify a correct omics type")
        
        img_rec = decoder(img_latent)
        img_h = head(img_rec.unsqueeze(1))
        img_pred = self.ae_classifier(img_h)
        
        return img_pred
        
    def forward(self, mrna, mirna, meth, img, stage='rec'):
        
        mrna_rec, _, mrna_mean,  mrna_logvar      = self.mrna_ae(mrna)
        mirna_rec, _, mirna_mean, mirna_logvar    = self.mirna_ae(mirna)
        meth_rec, _, meth_mean,  meth_logvar      = self.meth_ae(meth)
        img_rec, img_mean, img_logvar, img_latent = self.img_ae(img, return_z=True)

        mrna_rec_s = mrna_rec.unsqueeze(1)
        mirna_rec_s = mirna_rec.unsqueeze(1)
        meth_rec_s = meth_rec.unsqueeze(1)
        
        mrna_head = self.mrna_head(mrna_rec_s)
        mirna_head = self.mrna_head(mirna_rec_s)
        meth_head = self.mrna_head(meth_rec_s)

        mrna_pred  = self.ae_classifier(mrna_head)
        mirna_pred = self.ae_classifier(mirna_head)
        meth_pred  = self.ae_classifier(meth_head)
        
        if stage=='rec':
            
            return ((mrna_rec,  mrna_mean,  mrna_logvar,  mrna_pred),
                    (mirna_rec, mirna_mean, mirna_logvar, mirna_pred),
                    (meth_rec,  meth_mean,  meth_logvar,  meth_pred),
                    (img_rec,   img_mean,   img_logvar))
        
        elif stage=='pred':
            
            img_mrna_pred  = self.__img_predict(img_latent, 'mrna')
            img_mirna_pred = self.__img_predict(img_latent, 'mirna')
            img_meth_pred  = self.__img_predict(img_latent, 'meth')
       
            return ((mrna_rec,  mrna_mean,  mrna_logvar,  mrna_pred),
                    (mirna_rec, mirna_mean, mirna_logvar, mirna_pred),
                    (meth_rec,  meth_mean,  meth_logvar,  meth_pred),
                    (img_rec,   img_mean,   img_logvar,   img_mrna_pred, img_mirna_pred, img_meth_pred))
    
        else:
            raise NotImplementedError("Specify a valid stage")