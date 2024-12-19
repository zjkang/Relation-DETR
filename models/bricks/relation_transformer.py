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
    def __init__(self, decoder_layer, num_layers, num_classes, num_votes=16):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_relation_heads = decoder_layer.num_relation_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
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
        # self.position_relation_embedding = PositionRelationEmbedding(16, self.num_heads)
        # self.position_relation_embedding = MultiHeadCrossLayerHoughNetSpatialRelation(
        #     self.embed_dim, self.num_heads, self.num_votes)
        # self.position_relation_embedding = DualLayerBoxRelationEncoder(self.embed_dim, 16, self.num_heads)
        # self.position_relation_embedding = PositionRelationEmbeddingV2(16, self.num_heads)
        # self.position_relation_embedding = WeightedLayerBoxRelationEncoder(16, self.num_heads, num_layers=self.num_layers)
        self.position_relation_embedding = PositionRelationEmbeddingV3(16, self.num_relation_heads)

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
    ):
        outputs_classes, outputs_coords = [], []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        # currently pos_relation is the same as attn_mask
        pos_relation_mask = attn_mask  # fallback pos_relation to attn_mask
        relation_weights = None

        # 为normal attention创建专门的Q、K、V投影层
        # # NOTE: for changes not related to previous boxes, skip_relation is True
        if not skip_relation:
            # [batch_size, num_relation_heads, num_queries, num_queries]
            relation_weights = self.position_relation_embedding(reference_points)
            # if attn_mask is not None:
            #     pos_relation.masked_fill_(attn_mask, float("-inf"))

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
                self_attn_mask=pos_relation_mask,
                relation_weights=relation_weights,
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
            # if not skip_relation:
            #     src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
            #     tgt_boxes = output_coord
            #     pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
            #     if attn_mask is not None:
            #         pos_relation.masked_fill_(attn_mask, float("-inf"))


            # my possible relation embedding
            if not skip_relation:
                # src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
                tgt_boxes = output_coord
                relation_weights = self.position_relation_embedding(tgt_boxes)
                # if attn_mask is not None:
                #     pos_relation.masked_fill_(attn_mask, float("-inf"))

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
        n_relation_heads=4,  # 新增参数
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        self.num_relation_heads = n_relation_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 替换原有的self_attn
        self.self_attn = RelationAwareAttention(
            embed_dim, n_heads, n_relation_heads, dropout=dropout,batch_first=True)
        # # original attention: self attention
        # self.self_attn = nn.MultiheadAttention(
        #     embed_dim, n_heads, dropout=dropout, batch_first=True
        # )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention, skip it when using RelationAwareAttention
        # nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        # nn.init.xavier_uniform_(self.self_attn.out_proj.weight)

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
        relation_weights=None,
        self_attn_mask=None,
        key_padding_mask=None,
    ):
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            relation_weights=relation_weights,
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
# START: Relation Networks for Object Detection
# generate relation-weights to be used similar to position-embedding
class PositionRelationEmbeddingV3(nn.Module):
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
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 4,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        # # pos_proj: 将位置关系映射为标量权重
        # self.pos_proj = nn.Sequential(
        #     nn.Linear(embed_dim * 4, embed_dim * 2),  # *4 因为embed_dim后的维度是embed_dim*4
        #     nn.ReLU(inplace=True),
        #     nn.Linear(embed_dim * 2, 1)
        # )

    def forward(self, src_boxes: Tensor):
        tgt_boxes = src_boxes # target boxes are the same as source boxes
        # src_boxes: [batch_size, num_boxes1, 4]
        # tgt_boxes: [batch_size, num_boxes2, 4]
        torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
        with torch.no_grad():
            # [batch_size, num_queries, num_queries, 4]
            pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            # [batch_size, embed_dim*4, num_queries, num_queries]
            pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
        pos_embed = self.pos_proj(pos_embed)
        # [batch_size, num_heads, num_queries, num_queries]
        return pos_embed.clone()
# --------------------------------------------------------------------------------------------------


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
# START (V2.0): decoder-mul-relation with weighted layer-wise relation V2.0
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
#         # 原有的初始化
#         self.pos_proj = Conv2dNormActivation(
#             embed_dim * 7,  # 7 = 2(距离) + 2(尺度) + 2(方向) + 1(IoU)
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
#         # 添加多尺度权重
#         self.scale_weights = nn.Parameter(torch.ones(num_layers, 3))  # 3个尺度的权重
#         self.num_layers = num_layers
#         self.embed_dim = embed_dim
#         self.register_buffer('layer_idx', torch.zeros(1, dtype=torch.long))
#         self.eps = 1e-5
#         self.register_buffer('print_counter', torch.zeros(1, dtype=torch.long))
#         self.logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)

#     def forward(self, src_boxes: Tensor, layer_idx: Optional[int] = None):
#         tgt_boxes = src_boxes
#         torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
#         torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")

#         # 使用当前层索引或传入的层索引
#         curr_layer = self.layer_idx.item() if layer_idx is None else layer_idx
#         curr_layer = min(curr_layer, self.num_layers - 1)
#         layer_ratio = curr_layer / (self.num_layers - 1)

#         with torch.no_grad():
#             xy1, wh1 = src_boxes.split([2, 2], -1)
#             xy2, wh2 = tgt_boxes.split([2, 2], -1)

#             # 1. 计算相对距离
#             delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
#             relative_dist = delta_xy / (wh1.unsqueeze(-2) + self.eps)
#             # 使用log函数计算多尺度距离特征
#             # 局部注意力 - 强调近距离关系
#             local_distance = -torch.log(relative_dist + 1.0) # 距离越小,值越大
#             # 中等范围注意力
#             med_scale = 2.0 + layer_ratio * 3.0  # 随层数增加范围
#             medium_distance = -torch.log(relative_dist/med_scale + 1.0)
#             # 全局注意力保持不变
#             global_distance = torch.ones_like(local_distance) * 0.5

#             # 2. 计算尺度特征
#             wh_ratio = torch.log((wh1.unsqueeze(-2) + self.eps) / (wh2.unsqueeze(-3) + self.eps))
#             # 多尺度尺度特征
#             local_scale = -torch.log(torch.abs(wh_ratio) + 1.0)
#             medium_scale = -torch.log(torch.abs(wh_ratio)/med_scale + 1.0)
#             global_scale = torch.ones_like(local_scale) * 0.5

#             # 4. 方向特征
#             raw_delta_xy = xy1.unsqueeze(-2) - xy2.unsqueeze(-3)
#             theta = torch.atan2(raw_delta_xy[..., 1], raw_delta_xy[..., 0])
#             dir_embed = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

#             # 5. IoU特征
#             iou = box_iou(src_boxes, tgt_boxes)

#             # 6. 组合多尺度特征
#             # 局部特征
#             local_features = torch.cat([
#                 local_distance,          # [B, N, N, 2]
#                 local_scale,            # [B, N, N, 2]
#                 dir_embed,              # [B, N, N, 2]
#                 iou.unsqueeze(-1)       # [B, N, N, 1]
#             ], dim=-1)

#             # 中等范围特征
#             medium_features = torch.cat([
#                 medium_distance,         # [B, N, N, 2]
#                 medium_scale,           # [B, N, N, 2]
#                 dir_embed,              # [B, N, N, 2]
#                 iou.unsqueeze(-1)       # [B, N, N, 1]
#             ], dim=-1)

#             # 全局特征
#             global_features = torch.cat([
#                 global_distance,         # [B, N, N, 2]
#                 global_scale,           # [B, N, N, 2]
#                 torch.zeros_like(dir_embed),  # [B, N, N, 2]
#                 iou.unsqueeze(-1)       # [B, N, N, 1]
#             ], dim=-1)

#             # 7. 转换为位置编码
#             local_pos = self.pos_func(local_features).permute(0, 3, 1, 2)
#             medium_pos = self.pos_func(medium_features).permute(0, 3, 1, 2)
#             global_pos = self.pos_func(global_features).permute(0, 3, 1, 2)

#              # 堆叠所有尺度的特征 [B, 3, embed_dim * 7, N, N]
#             stacked_pos = torch.stack([local_pos, medium_pos, global_pos], dim=1)

#         # 应用可学习权重（确保梯度传播）
#         weights = F.softmax(self.scale_weights[curr_layer], dim=0)  # [3]
#         weights = weights.view(1, 3, 1, 1, 1)  # [1, 3, 1, 1, 1]

#         # 使用einsum进行加权求和
#         pos_embed = torch.einsum('bscdn,s->bcdn', stacked_pos, weights.squeeze())
#         # 或使用普通的矩阵运算
#         # pos_embed = (stacked_pos * weights).sum(dim=1)

#         # 10. 投影到最终的注意力权重
#         pos_embed = self.pos_proj(pos_embed)

#         # 更新层索引
#         if layer_idx is None:
#             self.layer_idx += 1
#             if self.layer_idx >= self.num_layers:
#                 self.layer_idx.zero_()

#         # 监控权重变化
#         if self.training and (dist.get_rank() == 0):
#             self.print_counter += 1
#             if self.print_counter % 100 == 0:  # 每100次迭代打印一次
#                 weights = F.softmax(self.scale_weights, dim=1).detach().cpu().numpy()
#                 self.logger.info("\nCurrent weights for each layer:")
#                 for i in range(self.num_layers):
#                     self.logger.info(f"Layer {i}: Local={weights[i,0]:.3f}, "
#                           f"Medium={weights[i,1]:.3f}, "
#                           f"Global={weights[i,2]:.3f}\n")

#         return pos_embed.clone()
# END: decoder-mul-relation with weighted layer-wise relation V3
# --------------------------------------------------------------------------------------------------


# class RelationAwareAttention(nn.Module):
#     def __init__(
#         self,
#         embed_dim, # 总的嵌入维度 (head_dim * num_heads)
#         num_heads,
#         num_relation_heads,
#         dropout=0.1,
#         batch_first=True
#     ):
#         super().__init__()
#         assert num_heads > num_relation_heads, "num_heads must be greater than num_relation_heads"
#         assert num_relation_heads > 0, "num_relation_heads must be greater than 0"

#         # 基本参数
#         self.embed_dim = embed_dim
#         self.num_normal_heads = num_heads - num_relation_heads
#         self.num_relation_heads = num_relation_heads
#         self.head_dim = embed_dim // num_heads
#         self.scaling = self.head_dim ** -0.5

#         normal_dim = self.head_dim * self.num_normal_heads
#         # 为normal attention创建专门的Q、K、V投影层
#         if self.num_normal_heads > 0:
#             normal_dim = self.head_dim * self.num_normal_heads
#             self.normal_q = nn.Linear(embed_dim, normal_dim)
#             self.normal_k = nn.Linear(embed_dim, normal_dim)
#             self.normal_v = nn.Linear(embed_dim, normal_dim)
#             self.normal_attention = nn.MultiheadAttention(
#                 normal_dim,
#                 self.num_normal_heads,
#                 dropout=dropout,
#                 batch_first=batch_first,
#                 add_bias_kv=False,
#                 add_zero_attn=False
#             )
#         else:
#             self.normal_attention = None

#         # 为relation heads准备的Q、K、V投影矩阵
#         self.relation_q = nn.Linear(embed_dim, num_relation_heads * self.head_dim)
#         self.relation_k = nn.Linear(embed_dim, num_relation_heads * self.head_dim)
#         self.relation_v = nn.Linear(embed_dim, num_relation_heads * self.head_dim)

#         # 输出投影和归一化
#         self.dropout = nn.Dropout(dropout)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.norm = nn.LayerNorm(embed_dim)

#         # 初始化权重
#         self.init_weights() 

#     def init_weights(self):
#         if self.num_normal_heads > 0:
#             # 初始化normal attention的投影层
#             nn.init.xavier_uniform_(self.normal_q.weight)
#             nn.init.xavier_uniform_(self.normal_k.weight)
#             nn.init.xavier_uniform_(self.normal_v.weight)
#             nn.init.constant_(self.normal_q.bias, 0.)
#             nn.init.constant_(self.normal_k.bias, 0.)
#             nn.init.constant_(self.normal_v.bias, 0.)
        
#         # initialize relation attention projections
#         nn.init.xavier_uniform_(self.relation_q.weight)
#         nn.init.xavier_uniform_(self.relation_k.weight)
#         nn.init.xavier_uniform_(self.relation_v.weight)
#         nn.init.xavier_uniform_(self.out_proj.weight)
        
#         # initialize biases to zero
#         nn.init.constant_(self.relation_q.bias, 0.)
#         nn.init.constant_(self.relation_k.bias, 0.)
#         nn.init.constant_(self.relation_v.bias, 0.)
#         nn.init.constant_(self.out_proj.bias, 0.)
        
#         # initialize normal attention if it exists
#         if self.normal_attention is not None:
#             nn.init.xavier_uniform_(self.normal_attention.in_proj_weight)
#             nn.init.xavier_uniform_(self.normal_attention.out_proj.weight)

#     def forward(
#         self,
#         query,                      # [batch_size, num_queries, embed_dim]
#         key,                        # [batch_size, num_queries, embed_dim]
#         value,                      # [batch_size, num_queries, embed_dim]
#         relation_weights,           # [batch_size, num_relation_heads, num_queries, num_queries]
#         attn_mask=None,            # [batch_size, num_queries, num_queries]
#         need_weights=False
#     ):
#         batch_size, num_queries = query.shape[:2]

#         # 1. 常规attention分支
#         if self.num_normal_heads > 0:
#             query_normal = self.normal_q(query)
#             key_normal = self.normal_k(key)
#             value_normal = self.normal_v(value)
            
#             normal_out = self.normal_attention(
#                 query_normal, key_normal, value_normal,
#                 attn_mask=attn_mask,
#                 need_weights=False
#             )[0]
#         else:
#             normal_out = None

#         # 2. Relation Network分支
#         # [batch_size, num_relation_heads, num_queries, num_queries]
#         if relation_weights is not None:
#             # 计算Q、K、V投影
#             q = self.relation_q(query).view(batch_size, -1, self.num_relation_heads, self.head_dim)
#             k = self.relation_k(key).view(batch_size, -1, self.num_relation_heads, self.head_dim)
#             v = self.relation_v(value).view(batch_size, -1, self.num_relation_heads, self.head_dim)

#             # 调整维度顺序 [batch_size, num_relation_heads, num_queries, head_dim]
#             q = q.transpose(1, 2)
#             k = k.transpose(1, 2)
#             v = v.transpose(1, 2)

#             # 计算attention scores
#             attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [B, H, N, N]

#             if attn_mask is not None:
#                 # 1. 检查mask的维度并调整
#                 if attn_mask.dim() == 2:
#                     # 如果mask是2D: [num_queries, num_queries]
#                     # 添加batch维度变成3D: [1, num_queries, num_queries]
#                     attn_mask = attn_mask.unsqueeze(0)
#                     # 扩展到正确的batch_size
#                     attn_mask = attn_mask.expand(batch_size, -1, -1)
                
#                 if attn_mask.dim() == 3:
#                     # 如果mask是3D: [batch_size, num_queries, num_queries]
#                     # 添加head维度变成4D: [batch_size, 1, num_queries, num_queries]
#                     attn_mask = attn_mask.unsqueeze(1)
                
#                 # 2. 扩展mask到所有attention heads
#                 attn_mask = attn_mask.expand(-1, self.num_relation_heads, -1, -1)
#                 # 3. 应用mask
#                 attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
#                 relation_weights = relation_weights.masked_fill(attn_mask, 0)

#             # 实现带关系权重的softmax
#             # 原始论文: w_ij = exp(w_G_ij * f_ij) / sum_k(exp(w_G_ik * f_ik))
#             # relation_weights已经是正确的维度 [B, H, N, N]
#             attn_weights = torch.exp(attn_weights) * relation_weights  # 直接相乘，不需要额外的维度调整
#             attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
#             attn_weights = self.dropout(attn_weights)

#             # 计算输出
#             relation_out = torch.matmul(attn_weights, v)  # [B, H, N, d]
#             relation_out = relation_out.transpose(1, 2).contiguous()  # [B, N, H, d]
#             # [batch_size, num_queries, num_relation_heads * head_dim]
#             relation_out = relation_out.view(batch_size, -1, self.num_relation_heads * self.head_dim)
#         else:
#             relation_out = None

#         if normal_out is None:
#             output = relation_out
#         elif relation_out is None:
#             output = normal_out
#         else:
#             # [B, N, (num_normal_heads + num_relation_heads) * head_dim]
#             # = [B, N, num_heads * head_dim]
#             # = [B, N, embed_dim]
#             output = torch.cat([normal_out, relation_out], dim=-1)

#         output = self.out_proj(output) # [B, N, embed_dim] -> [B, N, embed_dim]
#         output = self.norm(output) # [B, N, embed_dim]

#         return output, None


class RelationAwareAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_relation_heads,
        dropout=0.1,
        batch_first=True
    ):
        super().__init__()
        assert num_relation_heads > 0, "num_relation_heads must be greater than 0"
        assert num_relation_heads <= num_heads, "num_relation_heads cannot exceed num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_relation_heads = num_relation_heads
        self.num_normal_heads = num_heads - num_relation_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # 只保留一套投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query,
        key,
        value,
        relation_weights=None,
        attn_mask=None,
        need_weights=False,
        skip_relation=False
    ):
        batch_size, num_queries = query.shape[:2]
        
        # 投影 Q、K、V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头格式
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # 应用 mask（如果有）
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        # 根据 skip_relation 决定是否应用 relation_weights
        if not skip_relation and relation_weights is not None:
            if self.num_relation_heads == self.num_heads:
                # 所有heads都是relation heads
                attn_weights = torch.exp(attn_weights) * relation_weights
                attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                # 混合normal heads和relation heads
                # 对relation heads应用relation_weights
                relation_attn = torch.exp(attn_weights[:, -self.num_relation_heads:]) * relation_weights
                relation_attn = relation_attn / (relation_attn.sum(dim=-1, keepdim=True) + 1e-6)
                
                # normal heads使用普通的softmax
                normal_attn = F.softmax(attn_weights[:, :self.num_normal_heads], dim=-1)
                
                # 组合两种注意力权重
                attn_weights = torch.cat([normal_attn, relation_attn], dim=1)
        else:
            # 所有heads都使用普通的softmax
            attn_weights = F.softmax(attn_weights, dim=-1)

        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)
        output = self.norm(output)

        return output, None