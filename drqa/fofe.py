import torch
import torch.nn as nn

class fofe(nn.Module):
    def __init__(self, alpha):
        super(fofe, self).__init__()
        self.alpha = alpha
    def forward(self,x):
        length = x.size(1)
        #get the length of x and create matrix or vector
        matrix = torch.Tensor(x.size(0),length,length)
        output = torch.mul(matrix, x)
        return output


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
