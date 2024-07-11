# TagCLIP
This is the official repo of TagCLIP: Improving Discrimination Ability of Zero-Shot Semantic Segmentation.

## Overview
Contrastive Language-Image Pre-training (CLIP) has recently shown great promise in pixel-level zero-shot learning tasks. However, existing approaches utilizing CLIP's text and patch embeddings to generate semantic masks often misidentify input pixels from unseen classes, leading to confusion between novel classes and semantically similar ones. In this work, we propose a novel approach, **TagCLIP**~(**T**rusty-**a**ware **g**uided CLIP), to address this issue. We disentangle the ill-posed optimization problem into two parallel processes: semantic matching performed individually and reliability judgment for improving discrimination ability. Building on the idea of special tokens in language modeling representing sentence-level embeddings, we introduce a trusty token that enables distinguishing novel classes from known ones in prediction. To evaluate our approach, we conduct experiments on two benchmark datasets, PASCAL VOC 2012 and COCO-Stuff 164K. Our results show that TagCLIP improves the Intersection over Union (IoU) of unseen classes by 7.4% and 1.7%, respectively, with negligible overheads.

## Train
The training code will be released soon.