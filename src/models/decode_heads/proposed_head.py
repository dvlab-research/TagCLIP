import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import Linear, Dropout, LayerNorm
import numpy as np
import os
from typing import Optional
import math
from functools import partial
from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from timm.models.layers import trunc_normal_
from torch.nn.init import xavier_uniform_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy
from models.decode_heads.relationship_discriptor import build_relationship_discriptor
from models.decode_heads.attention import Attention

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

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class TPN_Decoder(nn.Module):
    def __init__(self, decoder_layers, disc_layer=None, norm=None):
        super().__init__()
        self.layers = decoder_layers
        # self.disc_layer = disc_layer
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, disc=False):
        outputs = [tgt]
        for mod in self.layers:
            outputs = mod(outputs[0], memory, tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask, disc=disc)

        # if not self.disc_layer:
        #     disc_output, disc_attn = self.disc_layer(output, memory, tgt_mask=tgt_mask,
        #                                             memory_mask=memory_mask,
        #                                             tgt_key_padding_mask=tgt_key_padding_mask,
        #                                             memory_key_padding_mask=memory_key_padding_mask)
            
        if self.norm is not None:
            outputs[0] = self.norm(outputs[0])

        # if discriminate output, attn, attn_disc
        # else: output, attn
        return outputs

class TPN_DecoderLayer_withAttn(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer_withAttn, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, disc=False):
        
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class TPN_DecoderLayer_withDiscHead(TransformerDecoderLayer):
    def __init__(self, num_classes, **kwargs):
        super(TPN_DecoderLayer_withDiscHead, self).__init__(**kwargs)
        del self.multihead_attn
        del self.self_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)
        self.disc_linear = Linear(num_classes, 2)

    def forward(self, tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask, disc=False):
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        # import pdb;pdb.set_trace()
        disc_attn = None
        if disc:
            disc_attn = self.disc_linear(attn2.transpose(-1, -2)).transpose(-1, -2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2, disc_attn

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        del self.self_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask, disc=False):

        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SingleMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation=F.relu, **factory_kwargs):
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.activation = activation
        
    def forward(self, x):
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return out
             
@HEADS.register_module()
class ProposedHead(BaseDecodeHead):
    def __init__(
            self,
            operator,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
            num_layers=3,
            embed_dims=768,
            num_heads=8,
            use_stages=1,
            use_proj=True,
            crop_train=False,
            rd_config=None,
            num_classes=None,
            discriminate=False, 
            eval_disc_weight=False,
            **kwargs,
    ):
        super(ProposedHead, self).__init__(
            in_channels=in_channels, num_classes=num_classes, **kwargs)

        assert use_stages == 1
        self.weights = None
        self.images = None
        self.image_paths = []
        self.image_size = img_size
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.relationship_discriptor = build_relationship_discriptor(**rd_config)

        nhead = num_heads
        dim = embed_dims

        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)

        if use_proj:
            self.input_proj = nn.Linear(self.in_channels, dim)
            trunc_normal_(self.input_proj.weight, std=.02)
        else:
            self.input_proj = nn.Identity()

        # norm layer
        self.proj_norm = nn.LayerNorm(dim) if use_proj else nn.Identity()

        # decoder layer
        decoder_layers = []
        assert num_layers == len(operator.split('_'))
        for s in operator.split('_'):
            if s == 'a':   
                decoder_layers.append(TPN_DecoderLayer_withAttn(d_model=dim, nhead=nhead, dim_feedforward=dim * 4))
            elif s == 'n':
                decoder_layers.append(TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4))
            # elif s == 'd':
            #     decoder_layers.append(TPN_DecoderLayer_withDiscHead(d_model=dim, nhead=nhead, dim_feedforward=dim * 4, num_classes=num_classes))
            else:
                raise
        
        # self.discriminate = discriminate
        self.decoder = TPN_Decoder(nn.ModuleList(decoder_layers))
        
        self.discriminate = discriminate
        if self.discriminate == 'rand':
            self.disc_token = nn.Parameter(torch.zeros(1, in_channels))
        elif self.discriminate in ['zero', True]:
            self.disc_token = nn.Parameter(torch.rand(1, in_channels))
        
        self.eval_disc_weight = eval_disc_weight

        delattr(self, 'conv_seg')
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs, img_metas, gt, train_cfg, self_training=False, st_mask=None):
        
        # import pdb;pdb.set_trace()
        seg_logits = self.forward(inputs)

        if self_training:
            pseudo_semantic_masks = seg_logits['pred_masks'].clone().detach().sigmoid() # [2, 20, 512, 512]
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(dim=1).unsqueeze(1)      # [2, 1, 512, 512]
            # generate pseudo labels for "transductive" setting
            # import pdb;pdb.set_trace()
            gt['gt_semantic_seg'][gt['gt_semantic_seg']==-1] = pseudo_semantic_seg[gt['gt_semantic_seg']==-1]
            gt['gt_semantic_seg'][gt['gt_semantic_seg']==-1] = 255
            losses = self.losses(seg_logits, gt)
            
        else:
            gt['gt_semantic_seg'][gt['gt_semantic_seg']==-1] = 255 # Empty set
            losses = self.losses(seg_logits, gt)

        return losses

    def forward_test(self, inputs, img, img_metas, test_cfg, self_training):
        return self.forward(inputs, img, img_metas, self_training)

    def semantic_inference(self, mask_pred, seen_idx, img, img_metas, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        mask_pred = self.weight_cls_pred(mask_pred, img, img_metas)
        # import pdb;pdb.set_trace()
        return mask_pred

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            # for a in zip(outputs_seg_masks[:-1])
            for a in outputs_seg_masks[:-1]
        ]

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def get_qs(self, q, cls):
        # q = [q.cls, q]
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        q1 = torch.einsum("bd,bcd->bcd", cls, q)
        q_ = torch.concat((q1, q), dim=-1)
        return q_

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, num_classes=None):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            # atm loss
            if isinstance(seg_label, dict):
                seg_label['gt_semantic_seg'] = seg_label['gt_semantic_seg'].squeeze(1)
                seg_label['gt_semantic_seg_unmasked'] = seg_label['gt_semantic_seg_unmasked'].squeeze(1)

                loss = self.loss_decode(
                    seg_logit,
                    seg_label,
                    ignore_index = self.ignore_index)

                loss['acc_seg'] = accuracy(seg_logit["pred_masks"], 
                                            seg_label['gt_semantic_seg'], 
                                            ignore_index=self.ignore_index)
                return loss
            
            else:
                # atm loss
                seg_label = seg_label.squeeze(1)

                loss = self.loss_decode(
                    seg_logit,
                    seg_label,
                    ignore_index = self.ignore_index)
                
                loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
                return loss

        
    def weight_cls_pred(self, pred, img, img_metas):
        # weight class probability by the weight of seen and unseen probability
        if self.discriminate:
            cls_pred = pred[:, :-1]
        else:
            cls_pred = pred

        if self.eval_disc_weight == 'seen':
            assert self.discriminate
            weight = pred[:, -1:]
            cls_pred[:, self.seen_idx] = cls_pred[:, self.seen_idx] * (1 - weight)
        elif self.eval_disc_weight in ['both', True]:
            assert self.discriminate
            weight = pred[:, -1:]
            cls_pred[:, self.seen_idx] = cls_pred[:, self.seen_idx] * (1 - weight)
            cls_pred[:, self.unseen_idx] = cls_pred[:, self.unseen_idx] * weight
        elif self.eval_disc_weight in ['unseen']:
            assert self.discriminate
            weight = pred[:, -1:]
            cls_pred[:, self.unseen_idx] = cls_pred[:, self.unseen_idx] * weight
        elif self.eval_disc_weight in [False]:
            pass
        elif self.eval_disc_weight.endswith('.npy'):
            assert self.discriminate
            weight = pred[:, -1:]
            if self.weights is not None:
                self.weights = torch.cat([self.weights, weight], axis=0)
                # self.images = torch.cat([self.images, img], axis=0)

            else:
                self.weights = weight
                # self.images = img

            self.image_paths.append(img_metas[0]['filename'])
            np.save(self.eval_disc_weight, self.weights.cpu().numpy())
            np.save('rebuttal/attention_map/trusty_weights/image_path.npy', self.image_paths)
            # np.save('rebuttal/attention_map/trusty_weights/images.npy', self.images.cpu().numpy())
            
            if len(self.weights) > 500:
                import pdb;pdb.set_trace()
            
        return cls_pred
                
    def forward(self, inputs_both, img=None, img_metas=None, self_training=None):
        ''' use inputs_both to produce'''
        # import pdb;pdb.set_trace()
        inputs = inputs_both[0][0][0]   # features:     [[1, 512, 32, 32]]
        cls_token = inputs_both[0][1]   # image logits: [1, 512]
        text_token = inputs_both[1]     # text logits:  [156, 512]
        if self.discriminate:           # disc logits:  [2, 512]     
            disc_token = self.disc_token.to(text_token.device)     
            text_token = torch.concat([text_token, disc_token], dim=0)
        
        x = self.d4_to_d3(inputs) if inputs.dim() > 3 else inputs
        bs = x.size()[0]

        q = self.relationship_discriptor(text_token, cls_token)
        q = q.transpose(0, 1)

        lateral = self.proj_norm(self.input_proj(x))

        q, attn = self.decoder(q, lateral.transpose(0, 1))
            
        attn = attn.transpose(-1, -2)

        attn = self.d3_to_d4(attn)
        size = attn.size()[-2:]

        outputs_seg_masks = F.interpolate(attn, size=size, mode='bilinear', align_corners=False)
        pred = F.interpolate(outputs_seg_masks,
                            size=(self.image_size, self.image_size),
                            mode='bilinear', align_corners=False)
        
        if self.discriminate:
            out = {
                "pred_masks": pred[:, :-1],
                "pred_disc": pred[:, -1:]
            }
        else:
            out = {"pred_masks": pred,}

        if not self.training:
            if self_training:
                out["pred"] = self.semantic_inference(pred, self.seen_idx, img, img_metas)
            else:
                out["pred"] = self.semantic_inference(pred, self.seen_idx, img, img_metas, 0.1)
            return out["pred"]

        return out

