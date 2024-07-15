import torch
import torch.nn as nn
from models.decode_heads.attention import Attention
from timm.models.layers import trunc_normal_


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def build_relationship_discriptor(type, **kwargs):
    RD_dict = {
        'UnlearnableRD': UnlearnableRD, 
        'LearnableRD': LearnableRD,
        'AttnRD': AttnRD,
        'SelfAttnRD': SelfAttnRD,
        'AttnCatRD': AttnCatRD,
    }
    assert type in RD_dict.keys(), f'Relationship Descriptor type {type} not implemented!'
    rd = RD_dict[type](**kwargs)
    return rd


def get_inputs(operator, q, s):
    inputs = []
    for v in operator.split('_'):
        if v == 'q':   
            inputs.append(q)
        elif v == 's':   
            inputs.append(s)
        elif v == 'qms':
            qms = torch.einsum("bcd,bcd->bcd", q, s)
            inputs.append(qms)
        elif v == 'qmscq':
            qms = torch.einsum("bcd,bcd->bcd", q, s)
            qmscq = torch.concat((qms, q), dim=-1)
            inputs.append(qmscq)
        elif v == 'qcs':
            qcs = torch.concat((q, s), dim=-1)
            inputs.append(qcs)
        else:
            raise NotImplementError(operator)
    return inputs


def get_input_dims(operator, dim):
    inputs = []
    for v in operator.split('_'):
        if v in ['q', 's', 'qms']:   
            inputs.append(dim)
        elif v in ['qmscq', 'qcs']:
            inputs.append(2*dim)
        else:
            raise NotImplementError(operator)
    return inputs


def set_input_proj(operator, dim):
    x = operator.split('_')[0] 
    if x in ['q', 's', 'qms']:
        q_proj = nn.Identity()
    elif x in ['qmscq', 'qcs']:
        q_proj = nn.Linear(2*dim, dim) 
        trunc_normal_init(q_proj, std=.02, bias=0)
    else:
        raise NotImplementError(operator)
    return q_proj
    

class UnlearnableRD(nn.Module):
    '''Unlearnable Relationship Discripter'''
    def __init__(self, operator, use_proj=True, proj_dims=512, dim=512, **kwargs):
        super().__init__()
        functions = {
            'qms_q': self.get_qms_q,
            's_q': self.get_s_q,
            'q': self.get_q,
        }
        assert operator in functions.keys(), f'Relationship Descriptor operator {operator} not implemented!'
        self.function = functions[operator]

        self.use_proj = use_proj
        if use_proj:
            self.q_proj = nn.Linear(proj_dims, dim) 
            trunc_normal_init(self.q_proj, std=.02, bias=0)
        else:
            self.q_proj = nn.Identity()
            assert proj_dims == dim, f'Please use projection to align the dimension {proj_dims} with {dim}!'

    def get_s_q(self, q, s):
        C, dim = q.shape
        bs, _ = s.shape
        # q: [C, dim] -> [bs, C, dim]
        q = q.expand(bs, -1, -1)
        # s: [bs, dim] -> [bs, C, dim]
        s = s.unsqueeze(1).expand(-1, C, -1)
        q_ = torch.concat((s, q), dim=-1)
        # q_: [bs, C, 2*dim]
        return q_

    def get_qms_q(self, q, s):
        C, dim = q.shape
        bs, _ = s.shape
        # q: [C, dim] -> [bs, C, dim]
        q = q.expand(bs, -1, -1)
        # q1: [bs, C, dim]
        q1 = torch.einsum("bd,bcd->bcd", s, q)
        q_ = torch.concat((q1, q), dim=-1)
        # q_: [bs, C, 2*dim]
        return q_

    def get_q(self, q, s):
        bs, _ = s.shape
        # q: [C, dim] -> [bs, C, dim]
        q_ = q.expand(bs, -1, -1).to(s.dtype)
        return q_

    def forward(self, text_token, cls_token):
        '''
        Args:
            cls_token:   [bs, dim]
            text_token:  [C, dim]
        Return:
            rd: [bs, C, dim]
        '''
        q = self.function(text_token, cls_token)
        q_ = self.q_proj(q)
        return q_


class LearnableRD(nn.Module):
    '''Learnable Relationship Discripter'''
    def __init__(self, operator, num_classes, use_proj=False, proj_dims=512, dim=512, **kwargs):
        super().__init__()
        if operator == 'q':
            self.q = nn.Embedding(num_classes, proj_dims)
            self.function = self.get_q
        else:
            raise NotImplementError(operator)

        self.use_proj = use_proj
        if use_proj:
            self.q_proj = nn.Linear(proj_dims, dim) 
            trunc_normal_init(self.q_proj, std=.02, bias=0)
        else:
            self.q_proj = nn.Identity()
            assert proj_dims == dim, f'Please use projection to align the dimension {proj_dims} with {dim}!'

    def get_q(self, q, s):
        bs, _ = s.shape
        # q: [C, dim] -> [bs, C, dim]
        q_ = self.q.weight.repeat(bs, 1, 1)
        return q_

    def forward(self, text_token, cls_token):
        '''
        Args:
            cls_token:   [bs, dim]
            text_token:  [C, dim]
        Return:
            rd: [bs, C, dim]
        '''
        q = self.function(text_token, cls_token)
        q_ = self.q_proj(q)
        return q_


class AttnRD(nn.Module):
    '''Attention Relationship Discripter'''
    def __init__(self, operator, dim, num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.attention = Attention(dim=dim, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.operator = operator
        self.proj = []
        for d in get_input_dims(operator, dim):
            proj = nn.Linear(d, dim) 
            trunc_normal_init(proj, std=.02, bias=0)
            self.proj.append(proj)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def function(self, q, s):
        inputs = get_inputs(self.operator, q, s)
        assert len(inputs) == len(self.proj)
        for i, p in enumerate(self.proj):
            p = p.to(inputs[i].device)
            inputs[i] = p(inputs[i])
        attn, _ = self.attention(inputs[0], inputs[1], inputs[2])
        attn = attn.transpose(0, 1)
        attn = inputs[0] + self.dropout(attn)
        attn = self.norm(attn)
        return attn

    def forward(self, text_token, cls_token):
        '''
        Args:
            cls_token:   [bs, dim]
            text_token:  [C, dim]
        Return:
            rd: [bs, C, dim]
        '''
        bs, _ = cls_token.shape
        C, dim = text_token.shape
        # text_token:  [C, dim] -> [bs, C, dim]
        text_token_ = text_token.expand(bs, -1, -1).to(cls_token.dtype)
        # cls_token:   [bs, dim] -> [bs, C, dim]
        cls_token_ = cls_token.unsqueeze(1).expand(-1, C, -1)
        # q: [C, bs, dim] -> [bs, C, dim]
        q = self.function(text_token_, cls_token_)
        return q


class SelfAttnRD(nn.Module):
    '''Self Attention Relationship Discripter'''
    def __init__(self, operator, num_heads, use_proj=False, 
                 proj_dims=512, dim=512, dropout=0.1, **kwargs):
        super().__init__()
        self.attention = Attention(dim=dim, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.operator = operator
        self.use_proj = use_proj
        self.input_proj = set_input_proj(self.operator, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def function(self, q, s):
        qms = torch.einsum("bcd, bcd->bcd", q, s)
        inputs = get_inputs(self.operator, q, s)
        assert len(inputs) == 1
        x = inputs[0]
        if self.use_proj:
            x = self.input_proj(x)
        attn, _ = self.attention(x, x, x)
        attn = attn.transpose(0, 1)
        attn = x + self.dropout(attn)
        attn = self.norm(attn)
        return attn

    def forward(self, text_token, cls_token):
        '''
        Args:
            cls_token:   [bs, dim]
            text_token:  [C, dim]
        Return:
            rd: [bs, C, dim]
        '''
        bs, _ = cls_token.shape
        C, dim = text_token.shape
        # text_token:  [C, dim] -> [bs, C, dim]
        text_token_ = text_token.expand(bs, -1, -1).to(cls_token.dtype)
        # cls_token:   [bs, dim] -> [bs, C, dim]
        cls_token_ = cls_token.unsqueeze(1).expand(-1, C, -1)
        # q: [C, bs, dim] -> [bs, C, dim]
        q = self.function(text_token_, cls_token_)
        return q


class AttnCatRD(nn.Module):
    '''Attention Concatnate Relationship Discripter'''
    def __init__(self, operator, num_heads, dim=512, dropout=0.1, **kwargs):
        super().__init__()
        self.operator = operator
        self.attention = Attention(dim=dim, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.q_proj = nn.Linear(2*dim, dim) 
        trunc_normal_init(self.q_proj.weight, std=.02, bias=0)
        self.input_proj = set_input_proj(self.operator, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def function(self, q, s):
        inputs = get_inputs(self.operator, q, s)
        attn, _ = self.attention(inputs[0], inputs[1], inputs[2])
        attn = attn.transpose(0, 1)
        attn = inputs[0] + self.dropout(attn)
        attn = self.norm(attn)
        # ([bs, C, dim], [bs, C, dim]) -> [bs, C, 2*dim] -> [bs, C, dim]
        q_ = self.q_proj(torch.concat((attn, q), dim=-1))
        return q_

    def forward(self, text_token, cls_token):
        '''
        Args:
            cls_token:   [bs, dim]
            text_token:  [C, dim]
        Return:
            rd: [bs, C, dim]
        '''
        bs, _ = cls_token.shape
        C, dim = text_token.shape
        # text_token:  [C, dim] -> [bs, C, dim]
        text_token_ = text_token.expand(bs, -1, -1).to(cls_token.dtype)
        # cls_token:   [bs, dim] -> [bs, C, dim]
        cls_token_ = cls_token.unsqueeze(1).expand(-1, C, -1)
        # q: [C, bs, dim] -> [bs, C, dim]
        q = self.function(text_token_, cls_token_)
        return q
