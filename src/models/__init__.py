from models.segmentor.zegclip import ZegCLIP
from models.segmentor.Proposed_segmentor import ProposedCLIPSegmentor

from models.backbone.text_encoder import CLIPTextEncoder
from models.backbone.img_encoder import CLIPVisionTransformer, VPTCLIPVisionTransformer
from models.decode_heads.decode_seg import ATMSingleHeadSeg

from models.losses.atm_loss import SegLossPlus

from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
from configs._base_.datasets.dataloader.voc12_bg import ZeroPascalVOCDataset20BG
from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
from configs._base_.datasets.dataloader.pascal_context import ZeroPascalContextDataset
