import copy
import functools
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional

from models.bricks.misc import Conv2dNormActivation
from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import get_sine_pos_embed
from util.misc import inverse_sigmoid

import torch.distributed as dist
import logging
import os


class RelationTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        hybrid_num_proposals: int = 900,
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.hybrid_num_proposals = hybrid_num_proposals
        self.num_classes = num_classes

        # model structure
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.hybrid_tgt_embed = nn.Embedding(hybrid_num_proposals, self.embed_dim)
        self.hybrid_class_head = nn.Linear(self.embed_dim, num_classes)
        self.hybrid_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        nn.init.normal_(self.hybrid_tgt_embed.weight)
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        nn.init.constant_(self.hybrid_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        noised_label_query=None,
        noised_box_query=None,
        attn_mask=None,
        time_dim=None,
    ):
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        topk, num_classes = self.two_stage_num_proposals, self.num_classes
        topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, num_classes))
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()
        target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)

        topk = self.hybrid_num_proposals if self.training else 0
        if self.training:
            # get hybrid classes and coordinates, target and reference points
            hybrid_enc_class = self.hybrid_class_head(output_memory)
            hybrid_enc_coord = self.hybrid_bbox_head(output_memory) + output_proposals
            hybrid_enc_coord = hybrid_enc_coord.sigmoid()
            topk_index = torch.topk(hybrid_enc_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
            hybrid_enc_class = hybrid_enc_class.gather(
                1, topk_index.expand(-1, -1, self.num_classes)
            )
            hybrid_enc_coord = hybrid_enc_coord.gather(1, topk_index.expand(-1, -1, 4))
            hybrid_reference_points = hybrid_enc_coord.detach()
            hybrid_target = self.hybrid_tgt_embed.weight.expand(
                multi_level_feats[0].shape[0], -1, -1
            )
        else:
            hybrid_enc_class = None
            hybrid_enc_coord = None

        num_denoising_queries = noised_label_query.shape[1] if noised_label_query is not None else 0
        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
            time_dim=time_dim,
            num_denoising_queries=num_denoising_queries,
        )

        if self.training:
            hybrid_classes, hybrid_coords = self.decoder(
                query=hybrid_target,
                value=memory,
                key_padding_mask=mask_flatten,
                reference_points=hybrid_reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                skip_relation=True,
            )
        else:
            hybrid_classes = hybrid_coords = None

        return (
            outputs_classes,
            outputs_coords,
            enc_outputs_class,
            enc_outputs_coord,
            hybrid_classes,
            hybrid_coords,
            hybrid_enc_class,
            hybrid_enc_coord,
        )


class RelationTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim
        self.memory_fusion = nn.Sequential(
            nn.Linear((num_layers + 1) * self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        reference_points,
        query_pos=None,
        query_key_padding_mask=None
    ):
        queries = [query]
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )
            queries.append(query)
        query = torch.cat(queries, -1)
        query = self.memory_fusion(query)
        return query


class RelationTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=query,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class RelationTransformerDecoder(nn.Module):
    def __init__(
            self,
            decoder_layer,
            num_layers,
            num_classes,
            num_queries=900,
            num_votes=16):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_votes = num_votes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        class_head = nn.Linear(self.embed_dim, num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        # relation embedding
        self.position_relation_embedding = PositionRelationEmbedding(16, self.num_heads)
        # self.position_relation_embedding = MultiHeadCrossLayerHoughNetSpatialRelation(
        #     self.embed_dim, self.num_heads, self.num_votes)
        # self.position_relation_embedding = DualLayerBoxRelationEncoder(self.embed_dim, 16, self.num_heads)
        # self.position_relation_embedding = PositionRelationEmbeddingV2(16, self.num_heads)
        # self.position_relation_embedding = WeightedLayerBoxRelationEncoder(16, self.num_heads, num_layers=self.num_layers)
        # self.position_relation_embedding = RankAwareRelationEncoder(
        #     16, self.num_heads, self.num_layers, self.num_queries)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initialize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
        skip_relation=False,
        time_dim=None,
        num_denoising_queries=None,
    ):
        outputs_classes, outputs_coords = [], []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        pos_relation = attn_mask  # fallback pos_relation to attn_mask
        # # NOTE: for changes not related to previous boxes, skip_relation is True
        # if not skip_relation:
        #     pos_relation = self.position_relation_embedding(reference_points, 0).flatten(0, 1)
        #     if attn_mask is not None:
        #         pos_relation.masked_fill_(attn_mask, float("-inf"))

        # query_predictions = {}

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            query_pos = query_pos * self.query_scale(query) if layer_idx != 0 else query_pos

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=pos_relation,
                time_dim=time_dim,
                num_denoising_queries=num_denoising_queries,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            # 偏移量预测
            output_coord = self.bbox_head[layer_idx](self.norm(query))
            output_coord = output_coord + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break


            # calculate position relation embedding
            # NOTE: prevent memory leak like denoising, or introduce totally separate groups?
            if not skip_relation:
                src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
                tgt_boxes = output_coord
                pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
                if attn_mask is not None:
                    pos_relation.masked_fill_(attn_mask, float("-inf"))

            # my possible relation embedding
            # if not skip_relation:
            #     src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
            #     tgt_boxes = output_coord
            #     pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
            #     if attn_mask is not None:
            #         pos_relation.masked_fill_(attn_mask, float("-inf"))

            # if not skip_relation:
            #     # src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
            #     # src_boxes, pred_logits=None, layer_idx=None
            #     # 获取attention mask和排序索引
            #     pos_relation, rank_indices = self.position_relation_embedding(
            #         output_coord, pred_logits=output_class, layer_idx=layer_idx)
            #     # 如果有排序索引，重排相关特征
            #     if rank_indices is not None:
            #         # 重排query
            #         query = torch.gather(
            #             query, 1,
            #             rank_indices.unsqueeze(-1).repeat(1, 1, query.shape[-1])
            #         )
            #         # 重排query_pos
            #         query_pos = torch.gather(
            #             query_pos, 1,
            #             rank_indices.unsqueeze(-1).repeat(1, 1, query_pos.shape[-1])
            #         )
            #         # 重排reference_points
            #         reference_points = torch.gather(
            #             reference_points, 1,
            #             rank_indices.unsqueeze(-1).repeat(1, 1, reference_points.shape[-1])
            #         )
            #     pos_relation = pos_relation.flatten(0, 1)
            #     if attn_mask is not None:
            #         pos_relation.masked_fill_(attn_mask, float("-inf"))


            # iterative bounding box refinement
            reference_points = inverse_sigmoid(reference_points.detach())
            reference_points = self.bbox_head[layer_idx](query) + reference_points
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)


        return outputs_classes, outputs_coords


class RelationTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        # 只为 denoising queries 添加时间处理, block time mlp
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 2)
        )

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        self_attn_mask=None,
        key_padding_mask=None,
        time_dim=None,
        num_denoising_queries=None,
    ):
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
            need_weights=False,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        if time_dim is not None and num_denoising_queries is not None and num_denoising_queries > 0:
            B = query.shape[0]  # 从query tensor获取batch size
            # 产生用于特征调制的 scale 和 shift 参数，维度是 [batch_size, embed_dim * 2]
            scale_shift = self.block_time_mlp(time_dim)
            # scale_shift 是从时间编码生成的调制参数，原始形状是 [batch_size, embed_dim * 2]
            # 使用 repeat_interleave 将其在第0维重复 num_denoising_queries 次，使其维度与特征维度匹配
            # 例如：如果原来是 [2, 512]，num_denoising_queries=200，则变成 400, 512]
            scale_shift = torch.repeat_interleave(scale_shift, num_denoising_queries, dim=0)
            # 将 scale_shift 张量沿着第1维分成两半
            scale, shift = scale_shift.chunk(2, dim=1) # scale, shift: [B * num_denoising_queries, embed_dim] 
            # 重塑 scale 和 shift 以匹配 denoising_queries 的维度
            scale = scale.view(B, num_denoising_queries, -1)  # [B, num_denoising_queries, embed_dim]
            shift = shift.view(B, num_denoising_queries, -1)  # [B, num_denoising_queries, embed_dim]

            # 分离 denoising 和 non-denoising 部分
            denoising_queries = query[:, :num_denoising_queries] # [B, num_denoising_queries, embed_dim]
            normal_queries = query[:, num_denoising_queries:] # [B, N-num_denoising_queries, embed_dim]
            # 只对 denoising 部分应用时间调制
            denoising_queries = denoising_queries * (scale + 1) + shift 
            # 重新组合
            query = torch.cat([denoising_queries, normal_queries], dim=1) # [B, N, embed_dim]

        # ffn
        query = self.forward_ffn(query)

        return query


def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

    return pos_embed


# --------------------------------------------------------------------------------------------------
# START: original implementation - ap: 52.2
class PositionRelationEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.,
        scale=100.,
        activation_layer=nn.ReLU,
        inplace=True,
    ):
        super().__init__()
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 4,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )

    def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        # src_boxes: [batch_size, num_boxes1, 4]
        # tgt_boxes: [batch_size, num_boxes2, 4]
        torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
        with torch.no_grad():
            pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
        pos_embed = self.pos_proj(pos_embed)

        return pos_embed.clone()
# END: original implementation - ap: 52.2
# --------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# START: only consider reference points (current box) on current layer using the same box relation encoding (delta_xy, delta_wh)
# layer0: learnable pos_embedding
class PositionRelationEmbeddingV2(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.,
        scale=100.,
        activation_layer=nn.ReLU,
        inplace=True,
    ):
        super().__init__()
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 4,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )

    def forward(self, src_boxes: Tensor):
        tgt_boxes = src_boxes # target boxes are the same as source boxes
        # src_boxes: [batch_size, num_boxes1, 4]
        # tgt_boxes: [batch_size, num_boxes2, 4]
        torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
        with torch.no_grad():
            pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
        pos_embed = self.pos_proj(pos_embed)

        return pos_embed.clone()
# --------------------------------------------------------------------------------------------------


class RankAwareRelationEncoder(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            num_layers,
            num_queries=900,
            temperature=10000.,
            scale=100.,
            activation_layer=nn.ReLU,
            inplace=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_queries = num_queries

        # 为每一层创建可学习的rank-aware relation query
        self.rank_aware_relation = nn.ModuleList([
            nn.Embedding(num_queries, embed_dim)  # 300是预设的最大query数量
            for _ in range(num_layers - 1)
        ])
        for m in self.rank_aware_relation.parameters():
            nn.init.zeros_(m)

        # 转换层
        self.pre_relation_trans = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(num_layers - 1)
        ])

        self.post_relation_trans = nn.ModuleList([
            nn.Linear(2 * embed_dim, embed_dim)  # 2*embed_dim因为要concat
            for _ in range(num_layers - 1)
        ])

        # 使用Conv2dNormActivation映射到attention mask
        # self.to_attention = nn.ModuleList([
        #     Conv2dNormActivation(
        #         embed_dim,  # 输入维度
        #         num_heads,  # 输出维度
        #         kernel_size=1,
        #         inplace=inplace,
        #         norm_layer=None,
        #         activation_layer=activation_layer,
        #     )
        #     for _ in range(num_layers)
        # ])
        self.to_attention = nn.ModuleList([
            nn.Sequential(
                # 先将[B, embed_dim, N, N]转换为[B, N, N, embed_dim]
                # 然后用Linear处理最后一维
                nn.Linear(embed_dim, num_heads),
                activation_layer(inplace=False)
            )
            for _ in range(num_layers - 1)
        ])
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim // 4,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )

    def forward(self, src_boxes, pred_logits=None, layer_idx=None):
        B, N = src_boxes.shape[:2]
        rank_indices = None

        with torch.no_grad():
            base_relation = box_rel_encoding(src_boxes, src_boxes)

            if pred_logits is not None and self.num_queries is not None:
                num_normal_queries = self.num_queries
                num_denoising = N - num_normal_queries

                # 只对normal queries部分计算置信度和排序
                normal_scores = pred_logits[:, num_denoising:].sigmoid().max(-1)[0]
                normal_rank_indices = normal_scores.argsort(dim=1, descending=True)

                # 构建完整的排序索引
                rank_indices = torch.arange(N, device=src_boxes.device)
                rank_indices = rank_indices.unsqueeze(0).expand(B, -1).clone()

                # 安全地更新normal queries部分
                normal_indices = normal_rank_indices + num_denoising
                rank_indices[:, num_denoising:] = normal_indices

                # 分块处理relation matrix
                normal_relation = base_relation[:, num_denoising:, num_denoising:]

                # 只重排normal queries部分
                normal_relation = torch.gather(
                    normal_relation, 1,
                    normal_rank_indices.unsqueeze(-1).unsqueeze(-1).repeat(
                        1, 1, N-num_denoising, base_relation.shape[-1])
                )
                normal_relation = torch.gather(
                    normal_relation, 2,
                    normal_rank_indices.unsqueeze(1).unsqueeze(-1).repeat(
                        1, N-num_denoising, 1, base_relation.shape[-1])
                )

                # 更新base_relation中的对应部分
                base_relation[:, num_denoising:, num_denoising:] = normal_relation

        base_relation = self.pos_func(base_relation) # [B, N, N, D]

        if pred_logits is not None:
            # 只为normal queries生成rank-aware relation
            normal_rank_relation = self.pre_relation_trans[layer_idx](
                self.rank_aware_relation[layer_idx].weight[:num_normal_queries].unsqueeze(0).expand(B, -1, -1)
            )  # [B, num_normal_queries, D]

            # 创建完整的rank_relation矩阵
            rank_relation = torch.zeros(
                B, N, self.embed_dim,
                device=src_boxes.device,
                dtype=src_boxes.dtype
            )

            # 只在normal queries部分填充rank-aware relation
            rank_relation[:, num_denoising:] = normal_rank_relation

            # 扩展rank_relation到N×N
            rank_relation = rank_relation.unsqueeze(2) + rank_relation.unsqueeze(1)  # [B, N, N, D]

            # 融合base_relation和rank_relation
            relation = torch.cat([base_relation, rank_relation], dim=-1)  # [B, N, N, 2D]
            relation = self.post_relation_trans[layer_idx](relation)  # [B, N, N, D]
        else:
            relation = base_relation

        # relation = relation.permute(0, 3, 1, 2)
        attention_mask = self.to_attention[layer_idx](relation).permute(0, 3, 1, 2) # [B, num_heads, N, N]


        return attention_mask.clone(), rank_indices



# --------------------------------------------------------------------------------------------------
# START: decoder-hough-only (decoder hough relation) - ap: 51.3
# class MultiHeadCrossLayerHoughNetSpatialRelation(nn.Module):
#     def __init__(
#             self,
#             embed_dim,
#             num_heads,
#             num_votes,
#             embed_pos_dim=16,
#             hidden_dim=32,#256,
#             temperature=10000.,
#             scale=100.,
#             activation_layer=nn.ReLU,
#             inplace=True):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_votes = num_votes
#         self.hidden_dim = hidden_dim

#         self.vote_generator = MLP(
#             self.embed_dim,
#             self.hidden_dim,
#             self.num_votes * 2,
#             1)

#         self.pos_proj = Conv2dNormActivation(
#             embed_pos_dim * 1, # (center_x, center_y, w, h, score) -> (score)
#             num_heads,
#             kernel_size=1,
#             inplace=inplace,
#             norm_layer=None,
#             activation_layer=activation_layer,
#         )
#         # This creates a partial function for generating sinusoidal position embeddings
#         self.pos_func = functools.partial(
#             get_sine_pos_embed,
#             num_pos_feats=embed_pos_dim, # embed_dim
#             temperature=temperature,
#             scale=scale,
#             exchange_xy=False,
#         )

#         # # 投票聚合器
#         # self.vote_aggregator = nn.Sequential(
#         #     nn.Conv2d(num_votes, hidden_dim, kernel_size=3, padding=1),
#         #     nn.ReLU(),
#         #     nn.Conv2d(hidden_dim, num_heads, kernel_size=1)
#         # )

#         # # 关系编码器 (现在为每个头生成单独的关系)
#         # self.relation_encoder = nn.Sequential(
#         #     nn.Linear(9*num_heads, hidden_dim),  # 9 * num_heads = 1 (aggregated vote) + 4 (current box) + 4 (previous box)
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim, num_heads)
#         # )

#     def forward(self, queries, current_ref_points, prev_ref_points):
#         """
#         Args:
#         - queries: Tensor of shape [batch_size, num_queries, embed_dim]
#         - current_ref_points: Tensor of shape [batch_size, num_queries, 4]
#         - prev_ref_points: Tensor of shape [batch_size, num_queries, 4]

#         Returns:
#         - attention_mask: Tensor of shape [batch_size, num_heads, num_queries, num_queries]
#         """
#         batch_size, num_queries, _ = queries.shape

#         # 生成投票
#         # votes = self.vote_generator(queries).view(batch_size, num_queries, self.num_votes, 2)
#         # current_ref = current_ref_points[:, :, :2].unsqueeze(2)  # (batch_size, num_queries, 1, 2)
#         # vote_positions = current_ref + votes  # (batch_size, num_queries, num_votes, 2)
#         vote_positions = current_ref_points[:, :, :2].unsqueeze(2) + \
#             self.vote_generator(queries).view(batch_size, num_queries, self.num_votes, 2)

#         # influence_map: [batch_size, num_queries, num_queries]
#         influence_map = self.create_influence_map(vote_positions, current_ref_points)
#         # del vote_positions

#         # temporarily disable
#         # with torch.no_grad():
#         #     # pos_embed: [batch_size, num_boxes1, num_boxes2, 4]
#         #     pos_embed = box_rel_encoding(prev_ref_points, current_ref_points)


#         # 将 influence_map 扩展一个维度以匹配 pos_embed 的形状
#         # influence_map_expanded = influence_map.unsqueeze(-1)  # [batch_size, num_queries, num_queries, 1]
#         # 拼接 influence_map 和 pos_embed
#         # fused_embed = torch.cat([influence_map.unsqueeze(-1), pos_embed], dim=-1)  # [batch_size, num_queries, num_queries, 5]

#         # # 如果需要，可以通过一个线性层来融合这些特征
#         # fusion_layer = nn.Linear(5, output_dim)  # 创建一个线性层来融合特征
#         # fused_embed = fusion_layer(fused_embed)  # [batch_size, num_queries, num_queries, output_dim]
#         # 如果需要用于注意力机制，可能还需要重塑维度
#         # fused_embed = fused_embed.permute(0, 3, 1, 2)  # [batch_size, output_dim, num_queries, num_queries]

#         # fused_embed: [batch_size, 5 * embed_dim, num_boxes1, num_boxes2]
#         # fused_embed = self.pos_func(fused_embed).permute(0, 3, 1, 2)
#         # fused_embed: [batch_size, num_heads, num_boxes1, num_boxes2]
#         # fused_embed = self.pos_proj(fused_embed)

#         # # 直接计算 fused_embed，不存储中间结果
#         # fused_embed = self.pos_proj(
#         #     self.pos_func(torch.cat(
#         #         [influence_map.unsqueeze(-1), pos_embed], dim=-1)).permute(0, 3, 1, 2))


#         fused_embed = self.pos_proj(
#             self.pos_func(torch.cat(
#                 [influence_map.unsqueeze(-1)], dim=-1)).permute(0, 3, 1, 2))
#         # del influence_map, pos_embed

#         return fused_embed.clone()


#     def create_influence_map(self, vote_positions, reference_points):
#         """
#         创建影响图

#         Args:
#         - vote_positions: [batch_size, num_queries, num_votes, 2]
#         - reference_points: [batch_size, num_queries, 4] (x_center, y_center, width, height)

#         Returns:
#         - influence_map: [batch_size, num_queries, num_queries]
#         """
#         batch_size, num_queries, num_votes, _ = vote_positions.shape

#         # 提取参考点的中心坐标
#         reference_centers = reference_points[:, :, :2]  # [batch_size, num_queries, 2]

#         # 将投票位置和参考中心点展平
#         vote_positions_flat = vote_positions.view(batch_size, num_queries * num_votes, 2)
#         reference_centers_flat = reference_centers.unsqueeze(2).expand(-1, -1, num_votes, -1).reshape(batch_size, num_queries * num_votes, 2)

#         # 计算每个投票到每个参考中心点的距离
#         distances = torch.cdist(vote_positions_flat, reference_centers, p=2)  # [batch_size, num_queries * num_votes, num_queries]

#         # 使用参考点的宽度和高度来调整 sigma
#         reference_sizes = reference_points[:, :, 2:]  # [batch_size, num_queries, 2] (width, height)
#         sigma = reference_sizes.mean(dim=-1, keepdim=True) / 2  # [batch_size, num_queries, 1]
#         sigma = sigma.repeat(1, num_votes, 1).view(batch_size, num_queries * num_votes, 1)  # [batch_size, num_queries * num_votes, 1]
#         # 使用自适应高斯核将距离转换为影响分数
#         influence_scores = torch.exp(-distances**2 / (2 * sigma**2))

#         # 重塑并求和得到最终的影响图
#         influence_map = influence_scores.view(batch_size, num_queries, num_votes, num_queries).sum(dim=2)

#         # 归一化影响图
#         influence_map = F.normalize(influence_map, p=1, dim=2)

#         # can improve from 41.0 to 41.8
#         influence_map = 1.0 - influence_map
#         # influence_map = -10000.0 * (1.0 - influence_map)
#         # [batch_size, num_queries, num_queries]
#         return influence_map
# END: decoder hough relation
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# # START: decoder-mul-relation: (dual layer box relation encoder)
# # V1: layer0: initial attn_mask; layer>=1: current layer & cross layer -ap: 52.1
# # V2: layer0: learnable pos_embedding; layer>=1: current layer & cross layer -ap: ?
# # V3: layer0: initial attn_mask; layer>=1: current layer -ap: 51.8
# # V4: layer0: learnable pos_embedding; layer>=1: current layer -ap: ?

def box_iou(boxes1, boxes2):
    """
    计算两组boxes的IoU
    boxes1, boxes2: [B, N, 4] (x_c, y_c, w, h)格式
    """
    # 转换到左上角和右下角格式
    boxes1_x0y0 = boxes1[..., :2] - boxes1[..., 2:] / 2
    boxes1_x1y1 = boxes1[..., :2] + boxes1[..., 2:] / 2
    boxes2_x0y0 = boxes2[..., :2] - boxes2[..., 2:] / 2
    boxes2_x1y1 = boxes2[..., :2] + boxes2[..., 2:] / 2

    # 计算交集区域的坐标
    intersect_mins = torch.max(boxes1_x0y0.unsqueeze(2), boxes2_x0y0.unsqueeze(1))
    intersect_maxs = torch.min(boxes1_x1y1.unsqueeze(2), boxes2_x1y1.unsqueeze(1))
    intersect_wh = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))

    # 计算交集面积
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # 计算两个boxes的面积
    area1 = (boxes1[..., 2] * boxes1[..., 3]).unsqueeze(2)
    area2 = (boxes2[..., 2] * boxes2[..., 3]).unsqueeze(1)

    # 计算并集面积
    union_area = area1 + area2 - intersect_area

    # 计算IoU
    iou = intersect_area / (union_area + 1e-6)
    return iou


# class DualLayerBoxRelationEncoder(nn.Module):
#     def __init__(
#             self,
#             d_model=256,
#             num_heads=8,
#             temperature=10000.,
#             scale=100.,
#             activation_layer=nn.ReLU,
#             inplace=True,):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads

#         # # 当前层的空间关系编码
#         # self.current_relation = nn.Sequential(
#         #     nn.Linear(9, d_model),
#         #     nn.ReLU(),
#         #     nn.Linear(d_model, num_heads)
#         # )

#         # # 层间关系编码
#         # self.inter_layer_relation = nn.Sequential(
#         #     nn.Linear(13, d_model),  # 13 = 9(当前层特征) + 4(与上一层的变化量)
#         #     nn.ReLU(),
#         #     nn.Linear(d_model, num_heads)
#         # )

#         self.pos_proj = Conv2dNormActivation(
#             d_model * 9,
#             num_heads,
#             kernel_size=1,
#             inplace=inplace,
#             norm_layer=None,
#             activation_layer=activation_layer,
#         )
#         self.pos_func = functools.partial(
#             get_sine_pos_embed,
#             num_pos_feats=d_model,
#             temperature=temperature,
#             scale=scale,
#             exchange_xy=False,
#         )

#     def compute_box_relation_features(self, ref_boxes):
#         """计算单层的box关系特征"""
#         B, N, _ = ref_boxes.shape

#         # 提取box信息
#         xy = ref_boxes[..., :2]
#         wh = ref_boxes[..., 2:]

#         # 计算中心点距离和归一化
#         delta_xy = xy.unsqueeze(1) - xy.unsqueeze(2)  # [B, N, N, 2]
#         avg_wh = (wh.unsqueeze(1) + wh.unsqueeze(2)) / 2  # [B, N, N, 2]
#         normalized_delta = delta_xy / (avg_wh + 1e-6)  # [B, N, N, 2]

#         # 宽高比例
#         wh_ratio = torch.log(wh.unsqueeze(1) / (wh.unsqueeze(2) + 1e-6))  # [B, N, N, 2]

#         # IoU
#         iou = box_iou(ref_boxes, ref_boxes)

#         # 方向编码
#         # rho = torch.norm(delta_xy, dim=-1)
#         theta = torch.atan2(delta_xy[..., 1], delta_xy[..., 0])
#         dir_sin = torch.sin(theta)
#         dir_cos = torch.cos(theta)
#         # 简单的角度量化
#         # theta = torch.atan2(delta_xy[..., 1], delta_xy[..., 0])
#         # angle_bins = quantize_angle(theta, num_bins=8)

#         # 相对宽高
#         relative_wh = wh.unsqueeze(1) - wh.unsqueeze(2)

#         # 组合特征
#         spatial_features = torch.cat([
#             normalized_delta,          # [B, N, N, 2]
#             wh_ratio,                  # [B, N, N, 2]
#             iou.unsqueeze(-1),        # [B, N, N, 1]
#             dir_sin.unsqueeze(-1),    # [B, N, N, 1]
#             dir_cos.unsqueeze(-1),    # [B, N, N, 1]
#             relative_wh               # [B, N, N, 2]
#         ], dim=-1)

#         return spatial_features


#     def compute_layer_transition(self, current_boxes, previous_boxes):
#         """计算两层之间的box变化"""
#         # 计算box变化量
#         delta_center = current_boxes[..., :2] - previous_boxes[..., :2]  # 中心点变化
#         delta_size = torch.log(current_boxes[..., 2:] / (previous_boxes[..., 2:] + 1e-6))  # 尺寸变化

#         # 将变化量扩展到N×N的pair矩阵
#         delta_center = delta_center.unsqueeze(2) - delta_center.unsqueeze(1)  # [B, N, N, 2]
#         delta_size = delta_size.unsqueeze(2) - delta_size.unsqueeze(1)  # [B, N, N, 2]

#         return torch.cat([delta_center, delta_size], dim=-1)  # [B, N, N, 4]

#     def forward(self, current_boxes, previous_boxes=None):
#         """
#         current_boxes: 当前层的参考框 [B, N, 4]
#         previous_boxes: 上一层的参考框 [B, N, 4]
#         """
#         # 检查输入
#         assert (current_boxes[..., 2:] > 0).all(), "Current boxes must have positive width/height"
#         if previous_boxes is not None:
#             assert (previous_boxes[..., 2:] > 0).all(), "Previous boxes must have positive width/height"

#         with torch.no_grad():
#             # 计算当前层的空间关系特征
#             current_features = self.compute_box_relation_features(current_boxes)

#             # if previous_boxes is None:
#             #     # 如果是第一层，只使用当前层的特征
#             #     relation_embed = self.current_relation(current_features)
#             # else:
#             # 计算层间变化
#             # transition_features = self.compute_layer_transition(current_boxes, previous_boxes)

#             # 组合当前层特征和层间变化特征
#             combined_features = torch.cat([current_features], dim=-1)
#             pos_embed = self.pos_func(combined_features).permute(0, 3, 1, 2)


#         pos_embed = self.pos_proj(pos_embed)

#         # pos_embed = self.pos_proj(self.pos_func(combined_features).permute(0, 3, 1, 2)) # [B, num_heads, N, N]

#         # pos_embed = self.inter_layer_relation(combined_features)
#         ## 调整维度顺序
#         # pos_embed = relation_embed.permute(0, 3, 1, 2)  # [B, num_heads, N, N]

#         return pos_embed.clone()
# # END: dual layer box relation encoder
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# START (V1.2): decoder-mul-relation with weighted layer-wise relation V2
# ignore local attention in higher layer
# 层级自适应权重:
#   添加可学习的层级权重参数 layer_weights
#   对距离、scale和方向特征分别应用不同的权重
#   权重通过 sigmoid 函数确保在 0-1 范围内
# spatial scale w/ layer: 距离、scale、方向
# 距离适应性: 较浅层关注近距离关系，深层允许更远距离的关系
# scale自适应: 浅层关注尺度变化，深层关注方向变化
# IoU阈值自适应: 浅层要求较高的IoU（0.7），深层允许较低的IoU（0.3）
# 方向自适应: 浅层关注局部方向，深层关注全局方向
# class WeightedLayerBoxRelationEncoder(nn.Module):
#     def __init__(
#         self,
#         embed_dim=256,
#         num_heads=8,
#         temperature=10000.,
#         scale=100.,
#         activation_layer=nn.ReLU,
#         inplace=True,
#         num_layers=6,
#     ):
#         super().__init__()
#         self.pos_proj = Conv2dNormActivation(
#             embed_dim * 7,  # 5 = 2(距离) + 2(尺度) + (2(方向)) + 1(IoU)
#             num_heads,
#             kernel_size=1,
#             inplace=inplace,
#             norm_layer=None,
#             activation_layer=activation_layer,
#         )
#         self.pos_func = functools.partial(
#             get_sine_pos_embed,
#             num_pos_feats=embed_dim,
#             temperature=temperature,
#             scale=scale,
#             exchange_xy=False,
#         )

#         # 添加层级权重参数
#         self.layer_weights = nn.Parameter(torch.ones(num_layers, 4))  # 3个权重分别对应距离、尺度和(方向),iou
#         self.num_layers = num_layers
#         self.embed_dim = embed_dim
#         self.register_buffer('layer_idx', torch.zeros(1, dtype=torch.long))
#         self.eps = 1e-5

#     # original:
#     # delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
#     # delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
#     # delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
#     # pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]
#     def forward(self, src_boxes: Tensor, layer_idx: Optional[int] = None):
#         tgt_boxes = src_boxes
#         torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
#         torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")

#         # 使用当前层索引或传入的层索引
#         curr_layer = self.layer_idx.item() if layer_idx is None else layer_idx
#         curr_layer = min(curr_layer, self.num_layers - 1)

#         # 获取当前层的权重
#         weights = torch.sigmoid(self.layer_weights[curr_layer])
#         distance_weight, scale_weight, direction_weight, iou_weight = weights[0], weights[1], weights[2], weights[3]


#         with torch.no_grad():
#             xy1, wh1 = src_boxes.split([2, 2], -1)
#             xy2, wh2 = tgt_boxes.split([2, 2], -1)

#             # 计算层级比例 [0,1]
#             layer_ratio = curr_layer / (self.num_layers - 1)

#             # 距离特征
#             delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
#             relative_dist = delta_xy / (wh1.unsqueeze(-2) + self.eps)
#             # 局部注意力
#             # 方案1：使用exp (适合强调近距离关系)
#             local_distance = torch.exp(-relative_dist) # [B, N, N, 2]
#             # 或者方案2：使用log (适合更均匀的距离关注)
#             # normalized_log = -torch.log(relative_dist + 1.0)
#             # local_distance = torch.sigmoid(normalized_log)  # 映射到(0,1)
#             # 全局注意力
#             global_distance = torch.ones_like(local_distance) * 0.5
#             # 混合局部和全局注意力
#             normalized_distance = (1 - layer_ratio) * local_distance + layer_ratio * global_distance
#             # 分别计算x和y方向的注意力: exp function
#             # normalized_distance = torch.exp(-delta_xy / (wh1.unsqueeze(-2) * distance_scale))  # [B, N, N, 2] -> [0,1]
#             # or 修改：使用负的log，这样距离越小，值越大
#             # normalized_distance = -torch.log(delta_xy / (wh1.unsqueeze(-2) * distance_scale) + 1.0)


#             # 尺度特征（使用log）
#             # 直接使用log比例，保留正负信息
#             wh_ratio = torch.log(
#                 (wh1.unsqueeze(-2) + self.eps) / (wh2.unsqueeze(-3) + self.eps))  # [B, N, N, 2]
#             # 局部注意力：使用对称的注意力函数
#             local_wh = torch.exp(-torch.abs(wh_ratio))  # (0,1]
#             # 或者使用高斯函数
#             # local_wh = torch.exp(-wh_ratio**2)  # (0,1]
#             # 全局注意力
#             global_wh = torch.ones_like(local_wh) * 0.5
#             # 混合
#             normalized_wh = (1 - layer_ratio) * local_wh + layer_ratio * global_wh
#             # # 使用exp将其映射到(0,1]范围
#             # scale_scale = 1.0 + curr_layer / self.num_layers
#             # normalized_wh = torch.exp(-wh_ratio / scale_scale)
#             # # (b) 使用log
#             # # delta_wh = torch.log((wh1.unsqueeze(-2) + self.eps) / (wh2.unsqueeze(-3) + self.eps))


#             # 方向特征（保持三角函数编码）
#             raw_delta_xy = xy1.unsqueeze(-2) - xy2.unsqueeze(-3)
#             theta = torch.atan2(raw_delta_xy[..., 1], raw_delta_xy[..., 0])
#             # 局部注意力 (使用三角函数，范围在[-1,1])
#             local_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
#             # 全局注意力 (使用0作为中性值，因为方向特征在[-1,1]范围内)
#             global_dir = torch.zeros_like(local_dir) # 0表示方向无关
#             # 混合
#             dir_embed = (1 - layer_ratio) * local_dir + layer_ratio * global_dir
#             # angle_scale = 1.0 + curr_layer / self.num_layers
#             # # 直接使用三角函数编码，不需要额外的权重
#             # dir_embed = torch.stack([
#             #     torch.cos(theta / angle_scale),  # 角度差异随层数变化
#             #     torch.sin(theta / angle_scale)
#             # ], dim=-1)


#             # IoU特征
#             # 局部注意力
#             local_iou = box_iou(src_boxes, tgt_boxes)  # [B, N, N] -> [0,1]
#             # 全局注意力
#             global_iou = torch.ones_like(local_iou) * 0.5
#             # 混合
#             iou = (1 - layer_ratio) * local_iou + layer_ratio * global_iou

#             # 将所有基础特征组合在一起
#             base_features = torch.cat([
#                 normalized_distance,
#                 normalized_wh,
#                 dir_embed,
#                 iou.unsqueeze(-1),
#             ], dim=-1)

#             # # 组合特征
#             # pos_embed = torch.cat([
#             #     normalized_distance * distance_weight,  # [B, N, N, 2]
#             #     normalized_wh * scale_weight,           # [B, N, N, 2]
#             #     dir_embed * direction_weight,           # [B, N, N, 2]
#             #     iou.unsqueeze(-1),                      # [B, N, N, 1]
#             # ], dim=-1)

#             # 转换为所需的格式
#             base_features = self.pos_func(base_features).permute(0, 3, 1, 2)

#         pos_embed = base_features * torch.cat([
#                 distance_weight.view(1, 1, 1, 1).expand(-1, self.embed_dim * 2, -1, -1),
#                 scale_weight.view(1, 1, 1, 1).expand(-1, self.embed_dim * 2, -1, -1),
#                 direction_weight.view(1, 1, 1, 1).expand(-1, self.embed_dim * 2, -1, -1),
#                 iou_weight.view(1, 1, 1, 1).expand(-1, self.embed_dim * 1, -1, -1)
#             ], dim=1)

#         pos_embed = self.pos_proj(pos_embed)

#         # 更新层索引
#         if layer_idx is None:
#             self.layer_idx += 1
#             if self.layer_idx >= self.num_layers:
#                 self.layer_idx.zero_()

#         return pos_embed.clone()
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# START (V1.4): decoder-mul-relation with weighted layer-wise relation V4
class WeightedLayerBoxRelationEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.,
        scale=100.,
        activation_layer=nn.ReLU,
        inplace=True,
        num_layers=6,
        num_classes=91,
    ):
        super().__init__()
        # 原有的初始化
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 5,  # 7 = 2(距离) + 2(尺度) + 2(方向) + 1(IoU)
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )

        # 添加多尺度权重
        initial_weights = torch.zeros(num_layers, 3)
        for i in range(num_layers):
            layer_ratio = i / (num_layers - 1)
            initial_weights[i] = torch.tensor([
                3.0 - layer_ratio * 2.0,  # 局部: 3.0->1.0
                0.8,                      # 中等: 0.8
                0.0 + layer_ratio * 2.0   # 全局: 0.0->2.0
            ])
        # log空间初始化
        self.scale_weights = nn.Parameter(torch.log(initial_weights))
        # 可选：添加小的随机扰动以打破对称性
        with torch.no_grad():
            self.scale_weights.data += torch.randn_like(self.scale_weights) * 0.02

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.eps = 1e-5
        self.register_buffer('layer_idx', torch.zeros(1, dtype=torch.long))
        self.register_buffer('print_counter', torch.zeros(1, dtype=torch.long))
        self.logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def forward(
            self, src_boxes: Tensor, layer_idx: Optional[int] = None, output_class: Optional[Tensor] = None):
        tgt_boxes = src_boxes
        torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")

        # 使用当前层索引或传入的层索引
        curr_layer = self.layer_idx.item() if layer_idx is None else layer_idx
        curr_layer = min(curr_layer, self.num_layers - 1)
        layer_ratio = curr_layer / (self.num_layers - 1)

        with torch.no_grad():
            xy1, wh1 = src_boxes.split([2, 2], -1)
            xy2, wh2 = tgt_boxes.split([2, 2], -1)

            # 计算相对距离
            delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
            relative_dist = delta_xy / (wh1.unsqueeze(-2) + self.eps)
            # 局部注意力 - 强调近距离关系
            local_distance = torch.exp(-relative_dist)
            # 中等范围注意力 - 使用高斯核
            med_scale = 2.0 + layer_ratio * 3.0  # 随层数增加范围
            medium_distance = torch.exp(-relative_dist**2 / (2 * med_scale**2))
            # 全局注意力 - 弱距离依赖
            global_distance = torch.ones_like(local_distance) * 0.5

            # 计算尺度特征
            wh_ratio = torch.log(
                (wh1.unsqueeze(-2) + self.eps) / (wh2.unsqueeze(-3) + self.eps))
            local_scale = torch.exp(-torch.abs(wh_ratio))
            medium_scale = torch.exp(-wh_ratio**2 / (2 * med_scale**2))
            global_scale = torch.ones_like(local_scale) * 0.5

            # 方向特征
            # raw_delta_xy = xy1.unsqueeze(-2) - xy2.unsqueeze(-3)
            # theta = torch.atan2(raw_delta_xy[..., 1], raw_delta_xy[..., 0])
            # dir_embed = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

            # IoU特征
            iou = box_iou(src_boxes, tgt_boxes)

            # 组合多尺度特征
            # 局部特征
            local_features = torch.cat([
                local_distance,          # [B, N, N, 2]
                local_scale,            # [B, N, N, 2]
                # dir_embed,              # [B, N, N, 2]
                iou.unsqueeze(-1)       # [B, N, N, 1]
            ], dim=-1)

            # 中等范围特征
            medium_features = torch.cat([
                medium_distance,         # [B, N, N, 2]
                medium_scale,           # [B, N, N, 2]
                # dir_embed,              # [B, N, N, 2]
                iou.unsqueeze(-1)       # [B, N, N, 1]
            ], dim=-1)

            # 全局特征
            global_features = torch.cat([
                global_distance,         # [B, N, N, 2]
                global_scale,           # [B, N, N, 2]
                # torch.zeros_like(dir_embed),  # [B, N, N, 2]
                iou.unsqueeze(-1)       # [B, N, N, 1]
            ], dim=-1)

            # 转换为位置编码
            local_pos = self.pos_func(local_features).permute(0, 3, 1, 2)
            medium_pos = self.pos_func(medium_features).permute(0, 3, 1, 2)
            global_pos = self.pos_func(global_features).permute(0, 3, 1, 2)

             # 堆叠所有尺度的特征 [B, 3, embed_dim * 7, N, N]
            stacked_pos = torch.stack([local_pos, medium_pos, global_pos], dim=1)

                # # 获取前景框的概率
                # foreground_probs = class_probs[:, :, :-1]  # [B, N, num_classes-1]
                # # 计算前景框的置信度
                # foreground_conf = foreground_probs.sum(dim=-1, keepdim=True)  # [B, N, 1]
                # # 获取背景框的概率
                # background_prob = class_probs[:, :, -1]  # [B, N]

        # 应用可学习权重（确保梯度传播）
        weights = F.softmax(self.scale_weights[curr_layer], dim=0).view(
            1, 3, 1, 1, 1)  #[3] -> [1, 3, 1, 1, 1]

        # 使用einsum进行加权求和,投影到最终的注意力权重
        pos_embed = self.pos_proj(
            torch.einsum('bscdn,s->bcdn', stacked_pos, weights.squeeze())
        )
        # 或使用普通的矩阵运算,
        # pos_embed = (stacked_pos * weights).sum(dim=1)

        # 更新层索引
        if layer_idx is None:
            self.layer_idx += 1
            if self.layer_idx >= self.num_layers:
                self.layer_idx.zero_()

        # 监控权重变化
        with torch.no_grad():
            if self.training and (self.rank == 0):
                self.print_counter += 1
                if self.print_counter % 1000 == 0:  # 每1000次迭代打印一次
                    self.print_counter.zero_()
                    weights = F.softmax(self.scale_weights, dim=1).detach().cpu().numpy()
                    self.logger.info("\nCurrent weights for each layer:")
                    for i in range(self.num_layers):
                        self.logger.info(f"Layer {i}: Local={weights[i,0]:.3f}, "
                            f"Medium={weights[i,1]:.3f}, "
                            f"Global={weights[i,2]:.3f}")

                    # output class
                    if output_class is not None:
                        # # output_class: [B, N, num_classes]
                        # # output_class = output_class.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                        # class_probs = output_class.softmax(dim=-1)  # [B, N, num_classes]
                        # # 获取每个预测的最大概率对应的类别索引
                        # max_class = class_probs.argmax(dim=-1)  # [B, N]
                        # 计算背景框的数量

                        class_probs = output_class.detach().softmax(dim=-1)  # [B, N, num_classes]
                        max_class = class_probs.argmax(dim=-1)  # [B, N]
                        num_background = (max_class == self.num_classes - 1).sum(dim=-1)  # [B]
                        avg_background = num_background.float().mean().item()
                        # self.print_counter.zero_()
                        # 记录当前层和背景数量
                        self.logger.info(f"Layer {curr_layer}: Average number of background predictions: {avg_background:.1f}")
                        # 可选：记录更详细的分布信息
                        # 计算每个类别的预测数量
                        class_counts = torch.bincount(
                            max_class.view(-1),
                            minlength=self.num_classes
                        ).float() / max_class.numel()

                        # 计算前景vs背景的比例
                        foreground_ratio = 1 - class_counts[-1].item()
                        self.logger.info(f"Foreground ratio: {foreground_ratio:.3f}")

                        # 可选：记录top-k最常预测的类别
                        # top_k = 5
                        # values, indices = class_counts[:-1].topk(top_k)  # 排除背景类
                        # self.logger.info("Top {} predicted classes:".format(top_k))
                        # for idx, val in zip(indices, values):
                        #     self.logger.info(f"Class {idx}: {val:.3f}")

        return pos_embed.clone()

# 预期的权重变化：
# 浅层（例如layer 0-1）
# Local   : ~0.6-0.7  (关注局部细节)
# Medium  : ~0.2-0.3  (适度关注中等范围)
# Global  : ~0.1-0.2  (较少关注全局)
# 中层（例如layer 2-3）：
# Local   : ~0.4-0.5  (平衡局部信息)
# Medium  : ~0.3-0.4  (增加中等范围关注)
# Global  : ~0.2-0.3  (开始关注全局)
# 深层（例如layer 4-5）：
# Local   : ~0.2-0.3  (减少局部关注)
# Medium  : ~0.3-0.4  (保持中等范围)
# Global  : ~0.4-0.5  (主要关注全局)
# END: decoder-mul-relation with weighted layer-wise relation V3
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# Weighted Layer-wise Relation V4: consider class-specific relation


# --------------------------------------------------------------------------------------------------





# --------------------------------------------------------------------------------------------------
# class WeightedLayerBoxRelationEncoder(nn.Module):
#     def __init__(
#             self,
#             embed_dim=256,
#             spatial_dim=256,
#             num_heads=8,
#             temperature=10000.,
#             scale=100.,
#             activation_layer=nn.ReLU,
#             inplace=True,):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.spatial_dim = spatial_dim
#         self.num_heads = num_heads

#         # # 当前层的空间关系编码
#         # self.current_relation = nn.Sequential(
#         #     nn.Linear(9, d_model),
#         #     nn.ReLU(),
#         #     nn.Linear(d_model, num_heads)
#         # )

#         # # 层间关系编码
#         # self.inter_layer_relation = nn.Sequential(
#         #     nn.Linear(13, d_model),  # 13 = 9(当前层特征) + 4(与上一层的变化量)
#         #     nn.ReLU(),
#         #     nn.Linear(d_model, num_heads)
#         # )

#         self.pos_proj = Conv2dNormActivation(
#             spatial_dim * 10, # 10 = 9(当前层特征) + 1(content编码)
#             num_heads,
#             kernel_size=1,
#             inplace=inplace,
#             norm_layer=None,
#             activation_layer=activation_layer,
#         )
#         self.pos_func = functools.partial(
#             get_sine_pos_embed,
#             num_pos_feats=spatial_dim,
#             temperature=temperature,
#             scale=scale,
#             exchange_xy=False,
#         )

#         self.hybrid_similarity = HybridSimilarity(embed_dim, spatial_dim)

#     def compute_box_relation_features(self, ref_boxes):
#         """计算单层的box关系特征"""
#         B, N, _ = ref_boxes.shape

#         # 提取box信息
#         xy = ref_boxes[..., :2]
#         wh = ref_boxes[..., 2:]

#         # 计算中心点距离和归一化
#         delta_xy = xy.unsqueeze(1) - xy.unsqueeze(2)  # [B, N, N, 2]
#         avg_wh = (wh.unsqueeze(1) + wh.unsqueeze(2)) / 2  # [B, N, N, 2]
#         normalized_delta = delta_xy / (avg_wh + 1e-6)  # [B, N, N, 2]

#         # 宽高比例
#         wh_ratio = torch.log(wh.unsqueeze(1) / (wh.unsqueeze(2) + 1e-6))  # [B, N, N, 2]

#         # IoU
#         iou = box_iou(ref_boxes, ref_boxes)

#         # 方向编码
#         # rho = torch.norm(delta_xy, dim=-1)
#         theta = torch.atan2(delta_xy[..., 1], delta_xy[..., 0])
#         dir_sin = torch.sin(theta)
#         dir_cos = torch.cos(theta)
#         # 简单的角度量化
#         # theta = torch.atan2(delta_xy[..., 1], delta_xy[..., 0])
#         # angle_bins = quantize_angle(theta, num_bins=8)

#         # 相对宽高
#         relative_wh = wh.unsqueeze(1) - wh.unsqueeze(2)

#         # 组合特征
#         spatial_features = torch.cat([
#             normalized_delta,          # [B, N, N, 2]
#             wh_ratio,                  # [B, N, N, 2]
#             iou.unsqueeze(-1),        # [B, N, N, 1]
#             dir_sin.unsqueeze(-1),    # [B, N, N, 1]
#             dir_cos.unsqueeze(-1),    # [B, N, N, 1]
#             relative_wh               # [B, N, N, 2]
#         ], dim=-1)

#         return spatial_features


#     def compute_layer_transition(self, current_boxes, previous_boxes):
#         """计算两层之间的box变化"""
#         # 计算box变化量
#         delta_center = current_boxes[..., :2] - previous_boxes[..., :2]  # 中心点变化
#         delta_size = torch.log(current_boxes[..., 2:] / (previous_boxes[..., 2:] + 1e-6))  # 尺寸变化

#         # 将变化量扩展到N×N的pair矩阵
#         delta_center = delta_center.unsqueeze(2) - delta_center.unsqueeze(1)  # [B, N, N, 2]
#         delta_size = delta_size.unsqueeze(2) - delta_size.unsqueeze(1)  # [B, N, N, 2]

#         return torch.cat([delta_center, delta_size], dim=-1)  # [B, N, N, 4]


#     def forward(self, queries, current_boxes, previous_boxes=None):
#         """
#         current_boxes: 当前层的参考框 [B, N, 4]
#         previous_boxes: 上一层的参考框 [B, N, 4]
#         """
#         # 检查输入
#         assert (current_boxes[..., 2:] > 0).all(), "Current boxes must have positive width/height"
#         if previous_boxes is not None:
#             assert (previous_boxes[..., 2:] > 0).all(), "Previous boxes must have positive width/height"

#         with torch.no_grad():
#             # 计算当前层的空间关系特征
#             current_features = self.compute_box_relation_features(current_boxes)

#             # if previous_boxes is None:
#             #     # 如果是第一层，只使用当前层的特征
#             #     relation_embed = self.current_relation(current_features)
#             # else:
#             # 计算层间变化
#             # transition_features = self.compute_layer_transition(current_boxes, previous_boxes)

#             # 组合当前层特征和层间变化特征
#             combined_features = torch.cat([current_features], dim=-1)
#             # [B, 9 * spatial_dim, N, N]
#             pos_embed = self.pos_func(combined_features).permute(0, 3, 1, 2)

#         # [B, spatial_dim, N, N]
#         content_embed = self.hybrid_similarity(queries).permute(0, 3, 1, 2)

#         # [B, num_heads, N, N]
#         pos_embed = self.pos_proj(torch.cat([pos_embed, content_embed], dim=1))

#         # pos_embed = self.pos_proj(self.pos_func(combined_features).permute(0, 3, 1, 2)) # [B, num_heads, N, N]

#         # pos_embed = self.inter_layer_relation(combined_features)
#         ## 调整维度顺序
#         # pos_embed = relation_embed.permute(0, 3, 1, 2)  # [B, num_heads, N, N]

#         return pos_embed.clone()


# --------------------------------------------------------------------------------------------------
# class HybridSimilarity(nn.Module):
#     def __init__(self, embed_dim, spatial_dim, num_bins=32):
#         super().__init__()
#         self.num_bins = num_bins

#         # 特征相似度编码
#         self.feat_sim_embed = nn.Embedding(num_bins, spatial_dim)

#         # 特征投影
#         self.feat_proj = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.ReLU()
#         )

#     def forward(self, queries):
#         B, N, C = queries.shape

#         # 1. 计算特征相似度
#         q_i = self.feat_proj(queries).unsqueeze(2)  # [B, N, 1, C]
#         q_j = self.feat_proj(queries).unsqueeze(1)  # [B, 1, N, C]

#         # 计算cosine相似度
#         cos_sim = F.cosine_similarity(q_i, q_j, dim=-1)  # [B, N, N]

#         # 2. 离散化相似度 [B, N, N]
#         sim_bins = (cos_sim * (self.num_bins-1)).long().clamp(0, self.num_bins-1)

#         # 3. 获取相似度embedding
#         sim_embed = self.feat_sim_embed(sim_bins)  # [B, N, N, spatial_dim]

#         return sim_embed
# END: dual layer box relation encoder
# ====================================================================================


# ====================================================================================
# START(In Progress): decoder-hough-space-relation: (decoder hough space relation)
# import torch
# import torch.nn as nn
# import math

# class HoughSpatialNet(nn.Module):
#     def __init__(self, input_dim, hough_dim, num_angle_bins=36):
#         super().__init__()
#         self.num_angle_bins = num_angle_bins

#         # 提取box特征的网络
#         self.box_encoder = nn.Sequential(
#             nn.Linear(input_dim, hough_dim),
#             nn.LayerNorm(hough_dim),
#             nn.ReLU()
#         )

#         # Hough空间特征提取网络
#         self.hough_encoder = nn.Sequential(
#             # 输入: 角度bin特征 + 距离特征 + 相对位置特征
#             nn.Linear(num_angle_bins + 3, hough_dim),
#             nn.LayerNorm(hough_dim),
#             nn.ReLU(),
#             nn.Linear(hough_dim, hough_dim)
#         )

#     def extract_hough_features(self, boxes_i, boxes_j):
#         """提取Hough空间特征
#         Args:
#             boxes_i: [B, N, 4] 格式为 (cx, cy, w, h)
#             boxes_j: [B, N, 4] 格式为 (cx, cy, w, h)
#         """
#         # 直接获取中心点
#         centers_i = boxes_i[..., :2]  # [B, N, 2] (cx, cy)
#         centers_j = boxes_j[..., :2]  # [B, N, 2] (cx, cy)

#         # 计算相对位置
#         delta = centers_j.unsqueeze(1) - centers_i.unsqueeze(2)  # [B, N, N, 2]

#         # 计算距离
#         distances = torch.norm(delta, dim=-1, keepdim=True)  # [B, N, N, 1]

#         # 计算角度
#         angles = torch.atan2(delta[..., 1], delta[..., 0])  # [B, N, N]

#         # 将角度转换为bin index
#         angles = (angles + math.pi) * self.num_angle_bins / (2 * math.pi)
#         angle_indices = angles.long() % self.num_angle_bins

#         # 生成角度的one-hot编码
#         angle_features = torch.zeros(
#             *angle_indices.shape, self.num_angle_bins,
#             device=angle_indices.device
#         )
#         angle_features.scatter_(-1, angle_indices.unsqueeze(-1), 1)

#         # 计算相对scale (面积比)
#         areas_i = boxes_i[..., 2] * boxes_i[..., 3]  # w * h
#         areas_j = boxes_j[..., 2] * boxes_j[..., 3]  # w * h
#         scale_ratio = torch.log(
#             areas_j.unsqueeze(1) / areas_i.unsqueeze(2)
#         ).unsqueeze(-1)  # [B, N, N, 1]

#         # 组合特征
#         hough_features = torch.cat([
#             angle_features,      # 角度bin特征
#             distances,          # 距离特征
#             scale_ratio,       # 相对scale特征
#             delta.norm(dim=-1, keepdim=True)  # 相对位置范数
#         ], dim=-1)

#         return hough_features

#     def forward(self, ref_boxes):
#         """
#         Args:
#             ref_boxes: [B, N, 4] tensor, 每个box格式为 (cx, cy, w, h)
#         """
#         B, N, _ = ref_boxes.size()

#         # 提取box基础特征
#         box_feats = self.box_encoder(ref_boxes)

#         # 提取Hough空间特征
#         hough_feats = self.extract_hough_features(ref_boxes, ref_boxes)
#         spatial_feats = self.hough_encoder(hough_feats)

#         return spatial_feats
# # END: decoder-hough-space-relation: (decoder hough space relation)
# ====================================================================================