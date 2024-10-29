import copy
import functools
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.bricks.misc import Conv2dNormActivation
from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import get_sine_pos_embed
from util.misc import inverse_sigmoid


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
        self.position_relation_embedding = MultiHeadCrossLayerHoughNetSpatialRelation(
            self.embed_dim, self.num_heads, self.num_votes)

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

        pos_relation = attn_mask  # fallback pos_relation to attn_mask
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
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
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

            if not skip_relation:
                src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
                tgt_boxes = output_coord
                pos_relation = self.position_relation_embedding(query, tgt_boxes, src_boxes).flatten(0, 1)
                if attn_mask is not None:
                    pos_relation.masked_fill_(attn_mask, float("-inf"))



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


# ------------------------------------------------------------
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
# -------------------------------------------------------------------


class MultiHeadCrossLayerHoughNetSpatialRelation(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            num_heads, 
            num_votes,
            hidden_dim=256,
            temperature=10000.,
            scale=100.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_votes = num_votes
        self.hidden_dim = hidden_dim
        
        # 1. 当前层的投票生成器
        self.curr_vote_generator = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_votes * 2)
        )
        
        # 2. 跨层特征融合
        self.cross_layer_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # 融合当前层和前一层的特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. 位置编码
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=hidden_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )
        
        # 4. 最终的多头注意力权重生成
        self.attention_proj = Conv2dNormActivation(
            hidden_dim * 2,  # 当前空间信息 + 跨层空间信息 #+ 语义信息
            num_heads,
            kernel_size=1,
            norm_layer=None,
            activation_layer=nn.ReLU
        )

    def forward(self, curr_queries, curr_ref_points, prev_queries=None, prev_ref_points=None):
        """
        Args:
            curr_queries: 当前层的queries [batch_size, num_queries, embed_dim]
            curr_ref_points: 当前层的参考点 [batch_size, num_queries, 4]
            prev_queries: 上一层的queries [batch_size, num_queries, embed_dim]
            prev_ref_points: 上一层的参考点 [batch_size, num_queries, 4]
        """
        batch_size, num_queries, _ = curr_queries.shape
        
        # 1. 当前层的空间关系
        curr_influence_map = self.compute_layer_influence(
            curr_queries, 
            curr_ref_points, 
            self.curr_vote_generator
        )
        
        # 2. 跨层空间关系
        if prev_queries is not None and prev_ref_points is not None:
            cross_layer_influence = self.compute_cross_layer_influence(
                curr_queries, curr_ref_points,
                prev_queries, prev_ref_points
            )
        else:
            cross_layer_influence = torch.zeros_like(curr_influence_map)
        
        # 3. 语义关系
        # semantic_map = self.create_semantic_map(curr_queries, prev_queries)
        
        # 4. 融合所有关系
        fused_relation = self.fuse_all_relations(
            curr_influence_map, 
            cross_layer_influence, 
            # semantic_map
        )
        
        return fused_relation.clone()

    def compute_layer_influence(self, queries, ref_points, vote_generator):
        """计算单层内的影响图"""
        batch_size, num_queries, _ = queries.shape
        
        # 生成投票
        vote_offsets = vote_generator(queries)
        vote_offsets = vote_offsets.view(batch_size, num_queries, self.num_votes, 2)
        
        # 使用box尺寸约束投票范围
        box_sizes = ref_points[:, :, 2:]
        vote_offsets = vote_offsets * box_sizes.unsqueeze(2) * 0.5
        
        # 生成投票位置
        vote_positions = ref_points[:, :, :2].unsqueeze(2) + vote_offsets
        
        return self.create_influence_map(vote_positions, ref_points)

    def compute_cross_layer_influence(self, curr_queries, curr_ref_points, 
                                    prev_queries, prev_ref_points):
        """计算跨层影响"""
        # 1. 计算特征相似度
        curr_feat = F.normalize(curr_queries, p=2, dim=-1)
        prev_feat = F.normalize(prev_queries, p=2, dim=-1)
        feat_sim = torch.bmm(curr_feat, prev_feat.transpose(1, 2))
        
        # 2. 计算空间关系
        pos_embed = box_rel_encoding(prev_ref_points, curr_ref_points)
        pos_embed = self.pos_func(pos_embed)
        
        # 3. 融合特征相似度和空间关系
        cross_influence = self.cross_layer_fusion(
            torch.cat([feat_sim.unsqueeze(-1), pos_embed], dim=-1)
        )
        
        return cross_influence

    # def create_semantic_map(self, curr_queries, prev_queries=None):
    #     """计算语义相似度图"""
    #     batch_size, num_queries, dim = curr_queries.shape
        
    #     # 当前层的语义相似度
    #     curr_norm = F.normalize(curr_queries, p=2, dim=-1)
    #     semantic_map = torch.bmm(curr_norm, curr_norm.transpose(1, 2))
        
    #     # 如果有前一层，加入跨层语义信息
    #     if prev_queries is not None:
    #         prev_norm = F.normalize(prev_queries, p=2, dim=-1)
    #         cross_semantic = torch.bmm(curr_norm, prev_norm.transpose(1, 2))
    #         semantic_map = semantic_map + cross_semantic
            
    #     return semantic_map

    def fuse_all_relations(self, curr_influence, cross_influence, semantic_map):
        """融合所有关系信息"""
        # 1. 编码各种关系
        curr_embed = self.pos_func(curr_influence.unsqueeze(-1))
        cross_embed = self.pos_func(cross_influence.unsqueeze(-1))
        semantic_embed = self.pos_func(semantic_map.unsqueeze(-1))
        
        # 2. 拼接并投影
        fused_embed = torch.cat([curr_embed, cross_embed, semantic_embed], dim=-1)
        fused_embed = fused_embed.permute(0, 3, 1, 2)
        attention_weights = self.attention_proj(fused_embed)
        
        return attention_weights
