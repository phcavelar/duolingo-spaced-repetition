import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HalfLifeRegression(nn.Module):
    def __init__(self,num_features, num_lexemes, base=2., h_min=15/60/24, h_max=9*30, p_min=0.0001, p_max=0.9999, epsilon=1e-6):
        super(HalfLifeRegression,self).__init__()
        self.linear = nn.Linear(num_features,1)
        self.lexeme_emb = nn.Embedding(num_lexemes,1)
        nn.init.constant_(self.lexeme_emb.weight,0.)
        self.base = nn.Parameter(torch.tensor(base),requires_grad=False)
        self.num_lexemes = num_lexemes
        
        self.h_min = nn.Parameter(torch.tensor(h_min),requires_grad=False)
        self.h_max = nn.Parameter(torch.tensor(h_max),requires_grad=False)
        self.p_min = nn.Parameter(torch.tensor(p_min),requires_grad=False)
        self.p_max = nn.Parameter(torch.tensor(p_max),requires_grad=False)
        self.eps = nn.Parameter(torch.tensor(epsilon),requires_grad=False)
    
    def forward(self, x, l, t):
        fx = self.linear(x)
        if self.num_lexemes>0:
            e = self.lexeme_emb(l)
            ex = fx + e
        half_life = torch.clamp(torch.pow(self.base,ex), self.h_min, self.h_max)
        prob = torch.clamp(torch.pow(2.,-t/(half_life+self.eps)), self.p_min, self.p_max)
        
        p_error_mask = 1-torch.isfinite(prob)
        h_error_mask = 1-torch.isfinite(half_life)
        assertion = torch.all(torch.isfinite(prob)) and torch.all(torch.isfinite(h_error_mask))
        error_mask = p_error_mask.squeeze()
        assert assertion, """NaN value in prob or half_life!
        x={x}
        fx={fx}
        e={e}
        ex={ex}
        l={l}
        h={h}
        t={t}
        p={p}
        """.format(x=x[error_mask],fx=x[error_mask],e=e[error_mask],ex=ex[error_mask],l=l[error_mask],h=half_life[error_mask],t=t[error_mask],p=prob[error_mask])
        return prob, half_life


class LogisticRegression(nn.Module):
    def __init__(self,num_features, num_lexemes, base=None, h_min=15/60/24, h_max=9*30, p_min=0.0001, p_max=0.9999, epsilon=1e-6):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(num_features,1)
        self.lexeme_emb = nn.Embedding(num_lexemes,1)
        nn.init.constant_(self.lexeme_emb.weight,0.)
        
        self.num_lexemes = num_lexemes
        
        self.h_min = nn.Parameter(torch.tensor(h_min),requires_grad=False)
        self.h_max = nn.Parameter(torch.tensor(h_max),requires_grad=False)
        self.p_min = nn.Parameter(torch.tensor(p_min),requires_grad=False)
        self.p_max = nn.Parameter(torch.tensor(p_max),requires_grad=False)
        self.eps = nn.Parameter(torch.tensor(epsilon),requires_grad=False)
    
    def forward(self, x, l, t):
        logit = self.linear(x)
        if self.num_lexemes>0: logit += self.lexeme_emb(l)
        prob = torch.clamp(torch.sigmoid(logit), self.p_min, self.p_max)
        half_life = torch.clamp(-t/(torch.log2(prob)+self.eps), self.h_min, self.h_max)
        return prob, half_life


