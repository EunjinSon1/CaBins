import sys

import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from models.clip.clip_base import *


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)

        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_Bins=256,score_map_dim=8):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(num_features+score_map_dim, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + features // 2, output_features=features // 2) #16
        self.up2 = UpSampleBN(skip_input=features // 2 + features // 4, output_features=features // 4) #8
        self.up3 = UpSampleBN(skip_input=features // 4 + features // 8, output_features=features // 8) #4

        self.conv3 = nn.Conv2d(features // 8, num_Bins, kernel_size=3, stride=1, padding=1)


    def forward(self, features):

        en4, en8, en16, en32, score_map = features[0], features[1], features[2], features[3], features[5]
        de_in = torch.concat([en32, score_map], dim=1)

        de32 = self.conv2(de_in)
        de16 = self.up1(de32, en16)
        de8 = self.up2(de16, en8)
        de4 = self.up3(de8, en4)

        de2 = F.interpolate(de4, size=[de4.size(2)*2 , de4.size(3)*2], mode='bilinear', align_corners=True)
        out = self.conv3(de2)

        return out, [de2, de4, de8, de16, de32]

class ComputeAdaptiveBins(nn.Module):
    def __init__(self,n_bins=256, min_val=0.001, max_val=10, embedding_dim=1024, dim_out=256):
        super(ComputeAdaptiveBins,self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val

        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 512),
                                       nn.LeakyReLU(),
                                       nn.Linear(512, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

        self.n_group = 8
        self.group_size = self.num_classes // self.n_group



    def forward(self, global_f):
        Bin = self.regressor(global_f)
        Bin = torch.relu(Bin)
        eps=0.1
        Bin = Bin + eps

        bin_widths_normed = Bin / Bin.sum(dim=1, keepdim=True)
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        n, dout = centers.size()
        centers_ = centers.view(n, dout, 1, 1)

        # generate 8 CaBins
        centers_group = centers.reshape(n, self.n_group, self.group_size)

        mean = torch.mean(centers_group, dim=-1, keepdim=True)
        diff = torch.abs(centers_group - mean) + 1e-4
        weighted_diff = torch.softmax((1/diff), dim=-1)
        CaBins = torch.sum(centers_group * weighted_diff, dim=-1) / torch.sum(weighted_diff, dim=-1)

        CaBins = CaBins.reshape(-1, n * self.n_group).squeeze(dim=0)

        CaBins = CaBins.tolist()
        CaBins = [str(round(element,4)) for element in CaBins]

        return bin_edges, centers_, CaBins


class Predict_Depthmap(nn.Module):
    def __init__(self,n_bins=256):
        super(Predict_Depthmap,self).__init__()
        self.n_bins = n_bins
        self.conv_out = nn.Sequential(nn.Conv2d(256, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, out, centers_):
        p_map = self.conv_out(out)
        pred = torch.sum(p_map * centers_, dim=1, keepdim=True)

        return pred


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):
        B, N, C = visual.shape

        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)

        return self.out_proj(x)


class TotalModel(nn.Module):
    def __init__(self, clip_model, gpu, max_depth, min_depth, n_bins):
        super(TotalModel, self).__init__()
        self.clip_model=clip_model
        self.gpu=gpu

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.n_bins = n_bins

        self.context_length=8
        self.token_embed_dim=512

        self.contexts = nn.Parameter(torch.randn(1, self.context_length, self.token_embed_dim))

        nn.init.trunc_normal_(self.contexts)

        self.context_decoder = ContextDecoder(visual_dim=512)
        self.gamma = nn.Parameter(torch.ones(512) * 1e-4)

        self.decoder = DecoderBN(num_Bins=256)
        self.computeAdaptiveBin = ComputeAdaptiveBins(embedding_dim=512, max_val=self.max_depth)

        self.predict_D = Predict_Depthmap(n_bins=256)

    def forward(self, img):
        # extract image embedding
        encoded_features = self.clip_model.encode_image(img)
        global_f, visual_embeddings = encoded_features[4]
        B, C, H, W = visual_embeddings.shape

        # generate adaptive bins
        bin_edges, centers_, CaBins = self.computeAdaptiveBin(global_f)

        texts = torch.cat([tokenize(c, context_length=30 - self.context_length) for c in CaBins]).to(self.gpu)
        texts = texts.reshape(B, len(CaBins)//B, 30 - self.context_length)
        learnable = self.contexts.to(self.gpu)
        learnable = learnable.expand(B, -1, -1)
        text_features = self.clip_model.encode_learnable_text(texts, learnable)

        visual_context = torch.cat([global_f.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],dim=2).permute(0, 2, 1).contiguous()
        text_diff = self.context_decoder(text_features, visual_context)
        text_features = text_features + self.gamma * text_diff

        visual = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_features, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual, text)
        encoded_features.append(score_map)

        # recover H/2 x W/2
        out, features = self.decoder(encoded_features)

        # predict probability map and depth map
        pred = self.predict_D(out, centers_)

        return bin_edges, pred