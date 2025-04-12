import torch
from einops import rearrange


class ARDecoder(torch.nn.Module):
    def __init__(self, indim, codebooknum, maxid):
        super(ARDecoder, self).__init__()
        self.codebooknum = codebooknum
        self.maxid = maxid
        outdim = codebooknum * maxid
        self.model = torch.nn.Linear(indim, outdim)

    # @torch.amp.autocast('cuda',enabled=False)
    def forward(self, x):
        # x = x.float()
        # [B,T,codebook_size*num_codebook]
        bs, t, _ = x.shape
        x = self.model(x)
        x = x.view(bs, t, self.maxid, self.codebooknum)
        x = rearrange(x, 'b t c n->b c t n')
        return x
