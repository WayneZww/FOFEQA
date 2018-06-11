import torch as torch
import torch.nn as nn
import torch.nn.functional as F

class fofe_conv1d(nn.Module):
    def __init__(self, channels, alpha=0.9, length=1, inverse=False):
        super(fofe_conv1d, self).__init__()
        self.alpha = alpha
        self.length = length
        self.channels = channels
        self.fofe_filter = self._init_filter(alpha, length, inverse)
        self.inverse_fofe_filter = self._init_filter(alpha, length, inverse)
        self.padding = (length - 1)//2
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(channels,channels*3,3,1,padding=length,
                        dilation=length, groups=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def _init_filter(self, alpha, length,inverse):
        self.fofe_filter = torch.Tensor(self.channels, 1, length)
        for i in range(length):
            if inverse :
                exponent = length - i
            else :
                exponent = i
            self.fofe_filter[:,:,i].fill_(torch.pow(alpha,exponent*torch.ones(self.channels,1)))

    def forward(self, x):        
        x = F.conv1d(x, self.fofe_filter, bias=0, stride=1, 
                        padding=self.padding, groups=self.channels)
        x = self.dilated_conv(x)
        return x


class fofe_linear(nn.Module):
    def __init__(self, channels, alpha): 
        super(fofe_linear, self).__init__()
        self.alpha = alpha
        self.linear = nn.Sequential(
            nn.Linear(channels,channels, bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        #get the length of x and create matrix or vector
        length = x.size(-1)
        matrix = torch.Tensor(x.size(0),length,length)
        x = torch.mul(matrix, x)
        output = self.linear(x)

        return output



class FOFE_Reader(nn.Module):
    def __init__(self, fofe_alpha, fofe_length, emb_dim, dilated):
        super(FOFE_NN_dilated, self).__init__()
        self.doc_fofe_conv = []
        for i in range(fofe_length):
            self.doc_fofe_conv.append(fofe_conv1d(emb_dim, fofe_alpha, i))
        self.doc_fofe_conv = nn.ModuleList(self.doc_fofe_conv)
        self.query_fofe = fofe_linear(emb_dim, fofe_alpha)
        self.emb_dim = emb_dim
        self.fnn = nn.Sequential(
            nn.Conv2d(emb_dim*4, emb_dim*4, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim*4, emb_dim*4, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim*4, emb_dim*4, 1, 1, bias=False),
            nn.ReLU(inplace=True)
            nn.Conv2d(emb_dim*4, emb_dim*2, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.s_conv = nn.Conv2d(emb_dim*2, 1, 1, 1, bias=False)
        self.e_conv = nn.Conv2d(emb_dim*2, 1, 1, 1, bias=False)
        
        

    def doc_fofe(self, x):
        fofe_out = []
        for fofe_layer in self.fofe_code:
            fofe_out.append(fofe_layer(x))
        fofe_out = torch.cat(fofe_out,1)

        return fofe_out

    def forward(self, query, document):
        query_fofe_code = self.query_fofe(query)
        doc_fofe_code = self.doc_fofe(document)
        query_doc = torch.Tensor(query_fofe_code.size(0), self.emb_dim*4,
                        query_fofe_code.size(2),query_fofe_code.size(3))
        query_doc[:,:self.emb_dim*3,:,:] = doc_fofe_code
        query_doc[:,self.emb_dim*3:self.emb_dim*4,:,:].fill_(query_fofe_code)
        x = self.fnn(query_doc)
        s_score = F.softmax(self.s_conv(x),dim=-2)
        e_score = F.softmax(self.e_conv(x),dim=-2)

        return s_score, e_score


"""class FOFE_CNN(nn.Module):
    def __init__(self, fofe_alpha, fofe_length, emb_dim, dilated):
        super(FOFE_DFNN, self).__init__()
        self.fofe_code = []
        for i in range(fofe_length):
            self.fofe_code.append(fofe_code1d(emb_dim, fofe_alpha, i))
        self.fofe_code = nn.ModuleList(self.fofe_conv)
        self.conv_layer = nn.Conv2d(emb_dim,emb_dim,(fofe_length, 3),1,1,0,1,bias=False)

    def fofo_conv(self, x):
        fofe_out = []
        for fofe_layer in self.fofe_code:
            fofe_out.append(fofe_layer(x))
        fofe_out = torch.cat(fofe_out,1)

        conv_out = []
        for conv_layer in self.conv_layer:
            conv_out.append(conv_layer(fofe_out))
        conv_out = torch.cat(conv_out,1)

        return conv_out

    def forward(self, query, document):
        query_fofe = self.fofe_conv(query)
        document_fofe = self.fofe_conv(document)
        
        return output


"""

"""
class FOFE_DFNN(nn.Module):
    def __init__(self, fofe_alpha):
        super(FOFE_DFNN, self).__init__()
        self.fofe = fofe(fofe_alpha)
        self.dfnn = nn.Sequential(
            nn.Linear(),
            nn.ReLU(inplace=True),
            nn.Linear(),
            nn.ReLU(inplace=True),
            nn.Linear(),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, query, document):
        query_fofe = self.fofe(query)
        document_fofe = self.fofe(document)
        #concat query's and document's fofe code
        dfnn_input = torch.cat((query_fofe,document_fofe),1)
        output = self.dfnn(dfnn_input)
        return output
"""
