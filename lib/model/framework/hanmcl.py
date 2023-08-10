import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import math

from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.framework.resnet import resnet101
import cv2


class HKA(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        return u * attn

class Attention(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = HKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
class HKA2(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv_spatial(x)
        attn = self.conv1(attn)
        return u * attn

class Attention2(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = HKA2(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
class _hANMCL(nn.Module):
    def __init__(self, classes, attention_type, rpn_reduce_dim, rcnn_reduce_dim, gamma, n_way=2, n_shot=5, pos_encoding=True):
        super(_hANMCL, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_way = n_way
        self.n_shot = n_shot
        self.attention_type = attention_type
        self.channel_gamma = gamma
        self.rpn_reduce_dim = rpn_reduce_dim
        self.rcnn_reduce_dim = rcnn_reduce_dim
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # pooling or align
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        # few shot rcnn head
        self.pool_feat_dim = 1024      #这个参数是什么 是1024
        self.rcnn_dim = 64
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))   #自适应池化   步长和人的大小都是随机的
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        dim_in = self.pool_feat_dim
        

        self.rpn_unary_layer = nn.Linear(400, 1)
        init.normal_(self.rpn_unary_layer.weight, std=0.01)
        init.constant_(self.rpn_unary_layer.bias, 0)
        self.rcnn_unary_layer = nn.Linear(49, 1)
        init.normal_(self.rcnn_unary_layer.weight, std=0.01)
        init.constant_(self.rcnn_unary_layer.bias, 0)

        # rpn_adapt_q1 q2 q3_layer都是一样的 这个的意义在那？ 难道是保存一下？
        self.rpn_adapt_q1_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_q1_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_q1_layer.bias, 0)
        self.rpn_adapt_k1_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_k1_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_k1_layer.bias, 0)
        
        self.rpn_adapt_q2_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_q2_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_q2_layer.bias, 0)
        self.rpn_adapt_k2_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_k2_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_k2_layer.bias, 0)

        self.rpn_adapt_q3_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_q3_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_q3_layer.bias, 0)
        self.rpn_adapt_k3_layer = nn.Linear(dim_in, rpn_reduce_dim)
        init.normal_(self.rpn_adapt_k3_layer.weight, std=0.01)
        init.constant_(self.rpn_adapt_k3_layer.bias, 0)
        
        
        self.rcnn_adapt_q1_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_q1_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_q1_layer.bias, 0)
        self.rcnn_adapt_k1_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_k1_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_k1_layer.bias, 0)

        self.rcnn_adapt_q2_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_q2_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_q2_layer.bias, 0)
        self.rcnn_adapt_k2_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_k2_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_k2_layer.bias, 0)

        
        self.rcnn_adapt_q3_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_q3_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_q3_layer.bias, 0)
        self.rcnn_adapt_k3_layer = nn.Linear(dim_in, rcnn_reduce_dim)
        init.normal_(self.rcnn_adapt_k3_layer.weight, std=0.01)
        init.constant_(self.rcnn_adapt_k3_layer.bias, 0)
        
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        if self.attention_type == 'concat':
            self.RCNN_rpn = _RPN(2048)
            self.rcnn_transform_layer = nn.Linear(2048, self.rcnn_dim)
        elif self.attention_type == 'product':
            self.RCNN_rpn = _RPN(1024)
            self.rcnn_transform_layer = nn.Linear(1024, self.rcnn_dim)
        
        self.output_score_layer = FFN(64* 49, dim_in)
        self.rcnn_transform_layer2 = nn.Linear(1024, 128)#线性层 直接获取 1024->128 就是全连接层
        # positional encoding
        self.pos_encoding = pos_encoding
        if pos_encoding:
            self.pos_encoding_layer = PositionalEncoding()
            self.rpn_pos_encoding_layer = PositionalEncoding(max_len=400)
        
        self.attention = Attention()
        self.attention2 = Attention2()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, support_ims, all_cls_gt_boxes=None):
        if self.training:
            self.num_of_rois = cfg.TRAIN.BATCH_SIZE
        else:
            self.num_of_rois = cfg.TEST.RPN_POST_NMS_TOP_N 
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        # 对获取的图像先backbone然后用attention进行进一步提取
        base_feat = self.RCNN_base(im_data)
        # attention模块是一个卷积型 是ham模块的第一次操作
        base_feat = self.attention(base_feat)
        base_feat = self.attention2(base_feat)
        if self.training:
            #训练状态 会得到 支持数据集 然后进行相应的操作
            # 注意张量的size b是什么batchsize   size 1024 20 20分别是什么
            support_ims = support_ims.view(-1, support_ims.size(2), support_ims.size(3), support_ims.size(4))
            support_feats = self.RCNN_base(support_ims)  # [B*2*shot, 1024, 20, 20]
            support_feats = self.attention(support_feats)
            support_feats = self.attention2(support_feats)
            #经过卷积会获得 get channel
            

            support_feats = support_feats.contiguous().view(-1, self.n_way*self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            #把support_feats变成了5维 且 第二维 way（class） * shot  正样本 有shot张 负样本有(cls-1)*shot
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :].contiguous()  # [B, shot, 1024, 20, 20]
            neg_support_feat = support_feats[:, self.n_shot:self.n_way*self.n_shot, :, :, :].contiguous()
            #怎么分配 pos 和neg的
            pos_support_feat_pooled = self.avgpool(pos_support_feat.view(-1, 1024, 20, 20))
            neg_support_feat_pooled = self.avgpool(neg_support_feat.view(-1, 1024, 20, 20))
            pos_support_feat_pooled = pos_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)  # [B, shot, 1024, 7, 7]
            neg_support_feat_pooled = neg_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)
            
            
        else:
            support_ims = support_ims.view(-1, support_ims.size(2),  support_ims.size(3),  support_ims.size(4)) 
            support_feats = self.RCNN_base(support_ims)
            support_feats = self.attention(support_feats)
            support_feats = self.attention2(support_feats)
            support_feats = support_feats.view(-1, self.n_shot, support_feats.size(1), support_feats.size(2), support_feats.size(3))
            
            pos_support_feat = support_feats[:, :self.n_shot, :, :, :]
            pos_support_feat_pooled = self.avgpool(pos_support_feat.view(-1, 1024, 20, 20))
            pos_support_feat_pooled = pos_support_feat_pooled.view(batch_size, self.n_shot, 1024, 7, 7)

        batch_size = pos_support_feat.size(0)
        feat_h = base_feat.size(2)
        feat_w = base_feat.size(3)
        support_mat = pos_support_feat.transpose(0, 1).view(self.n_shot, batch_size, 1024, -1).transpose(2, 3)
        query_mat = base_feat.view(batch_size, 1024, -1).transpose(1, 2)
        dense_support_feature = []
        #为什么要生成三层 因为权重不断变化所以内容不一样 每次backward对这三层更新的权重不一样 rpn_adapt_q1_layer 正好对应三层fc 全连阶层
        #q3的值 会和支持集 进行交互
        q1_matrix = self.rpn_adapt_q1_layer(query_mat)  # [B, hw, 256]
        q1_matrix = q1_matrix - q1_matrix.mean(1, keepdim=True)
        
        q2_matrix = self.rpn_adapt_q2_layer(query_mat)  # [B, hw, 256]
        q2_matrix = q2_matrix - q2_matrix.mean(1, keepdim=True)
        
        q3_matrix = self.rpn_adapt_q3_layer(query_mat)  # [B, hw, 256]
        q3_matrix = q3_matrix - q3_matrix.mean(1, keepdim=True)
        
        for i in range(self.n_shot):#n_shot到底是什么参数 shot是一次query 对应多少个支持集
            if self.pos_encoding:# 对支持特征进行位置编码的处理
                single_s_mat = self.rpn_pos_encoding_layer(support_mat[i])  # [B, 400, 1024]
            else:
                single_s_mat = support_mat[i]
            #k是key的意思 也就是 q-k
            k1_matrix = self.rpn_adapt_k1_layer(single_s_mat)  # [B, hw, 256]
            k1_matrix = k1_matrix - k1_matrix.mean(1, keepdim=True)
            k2_matrix = self.rpn_adapt_k2_layer(single_s_mat)  # [B, hw, 256]
            k2_matrix = k2_matrix - k2_matrix.mean(1, keepdim=True)
            
            k3_matrix = self.rpn_adapt_k3_layer(single_s_mat)  # [B, hw, 256]
            k3_matrix = k3_matrix - k3_matrix.mean(1, keepdim=True)
            #矩阵乘法 q1 q2相乘 并且归一化  这个不是一个size里面都是恒定的吗 是否有改进的空间 应该是有的 计算应该会快点
            support_adaptive_attention_weight1 = torch.bmm(q1_matrix, q2_matrix.transpose(1, 2)) / math.sqrt(self.rpn_reduce_dim) 
            support_adaptive_attention_weight1 = F.softmax(support_adaptive_attention_weight1, dim=2)
            #矩阵乘法 k1 k2相乘 并且归一化
            support_adaptive_attention_weight2 = torch.bmm(k1_matrix, k2_matrix.transpose(1, 2)) / math.sqrt(self.rpn_reduce_dim) 
            support_adaptive_attention_weight2 = F.softmax(support_adaptive_attention_weight2, dim=2)
            #矩阵乘法 q3 k3
            support_adaptive_attention_weight3 = torch.bmm(q3_matrix, k3_matrix.transpose(1, 2)) / math.sqrt(self.rpn_reduce_dim)
            support_adaptive_attention_weight3 = F.softmax(support_adaptive_attention_weight3, dim=2)
            #q1 q2的乘积和原始query进行乘法
            support_adaptive_attention_feature1 = torch.bmm(support_adaptive_attention_weight1, query_mat)  # [B, hw, 1024]

            #对获得的 结果进行 调整 使之能完成相加 rpn_unary_layer模块 用来干什么的
            unary_term = self.rpn_unary_layer(support_adaptive_attention_weight2)  # [n_roi, 49, 1]
            unary_term = F.softmax(unary_term, dim=1)
            #二和三融合 k1k2的值和 k3q3的值相加
            support_adaptive_attention_weight = support_adaptive_attention_weight3 + unary_term.transpose(1, 2)
            # 二和三融合的值 和原始key进行乘法
            support_adaptive_attention_feature2 = torch.bmm(support_adaptive_attention_weight, single_s_mat)  # [B, hw, 1024]

            #这两个结果concat一下 需要看一下 这个的维度 因为有很多个支持集
            support_adaptive_attention_feature = torch.cat([support_adaptive_attention_feature1,
                                                             support_adaptive_attention_feature2],dim=2)
            #在多次计算之后 dense_support_feature会发生怎么样的变化呢
            dense_support_feature += [support_adaptive_attention_feature]

        #dense_support_feature是由多组 query和key组成的 需要进行调整
        dense_support_feature = torch.stack(dense_support_feature, 0).mean(0)  # [B, hw, 1024]
        dense_support_feature = dense_support_feature.transpose(1, 2).contiguous().view(batch_size, 1024*2, feat_h, feat_w)
        # 压缩后的特征 主要帮助提出建议 后面也没啥作用
        if self.attention_type == 'concat':
            correlation_feat = dense_support_feature
        elif self.attention_type == 'product':#选择是否要进行别的操作
            correlation_feat = base_feat * dense_support_feature
        #获得图像建议 rcnn_rpn这种都是直接模块化出去的 用的时候直接引用就行
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(correlation_feat, im_info, gt_boxes, num_boxes)
        #这几个 返回值 都是什么意思
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        rois = Variable(rois)
        #选择align和pool作为输出  roi_align模块 输出是一个feature 需要查看他的维度
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # rcnn head  也就是ham模块 这里的输出 应该和 roi之前的输出是一致的
        if self.training:
            #把之前获取到的特征 全都输入到头中得出最后的结果  输入是 基础特征和建议融合后的特征 然后正样本比较
            bbox_pred, cls_prob, cls_score_all, correlation_feat2 = self.rcnn_head(pooled_feat, pos_support_feat_pooled)
            # neg feats 是由 池化的特征 和 支持集负样本的特征组成的 pooled主要由基础特征组成数据来源imdata
            _, neg_cls_prob, neg_cls_score_all, neg_correlation_feat2 = self.rcnn_head(pooled_feat, neg_support_feat_pooled)
            # 获取所有类的概率 正样本类和负样本类

            #连接正样本的概率和负样本的概率
            cls_prob = torch.cat([cls_prob, neg_cls_prob], dim=0)
            #cls_score_all是什么 neg_cls_score_all 他们的数据特征又是怎么样的 得看rcnn_head这个函数
            cls_score_all = torch.cat([cls_score_all, neg_cls_score_all], dim=0)
            neg_rois_label = torch.zeros_like(rois_label)
            rois_label = torch.cat([rois_label, neg_rois_label], dim=0)
            fg_inds = (rois_label == 1).nonzero().squeeze(-1)
            # # 将正样本和负样本的相关特征进行堆叠和维度转换  这个是对比学习吗
            roi = torch.stack((correlation_feat2[fg_inds],neg_correlation_feat2[fg_inds]),dim=0)
            roi = roi.permute(1,0,2)

        else:
            bbox_pred, cls_prob, cls_score_all, correlation_feat2 = self.rcnn_head(pooled_feat, pos_support_feat_pooled)

        # losses
        if self.training:
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            fg_inds = (rois_label == 1).nonzero().squeeze(-1)
            bg_inds = (rois_label == 0).nonzero().squeeze(-1)
            cls_score_softmax = torch.nn.functional.softmax(cls_score_all, dim=1)
            bg_cls_score_softmax = cls_score_softmax[bg_inds, :]
            bg_num_0 = max(1, min(fg_inds.shape[0] * 2, int(rois_label.shape[0] * 0.25)))
            bg_num_1 = max(1, min(fg_inds.shape[0], bg_num_0))
            _sorted, sorted_bg_inds = torch.sort(bg_cls_score_softmax[:, 1], descending=True)
            real_bg_inds = bg_inds[sorted_bg_inds]  # sort the real_bg_inds
            real_bg_topk_inds_0 = real_bg_inds[real_bg_inds < int(rois_label.shape[0] * 0.5)][:bg_num_0]  # pos support
            real_bg_topk_inds_1 = real_bg_inds[real_bg_inds >= int(rois_label.shape[0] * 0.5)][:bg_num_1]  # neg_support
            topk_inds = torch.cat([fg_inds, real_bg_topk_inds_0, real_bg_topk_inds_1], dim=0)
            RCNN_loss_cls = F.cross_entropy(cls_score_all[topk_inds], rois_label[topk_inds])
        else:
            RCNN_loss_cls = 0
            RCNN_loss_bbox = 0
            rois_label2=0
            roi=[]
            roi_temp=0
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, roi#, roi_temp

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def rcnn_head(self, pooled_feat, support_feat):
        # box regression
        bbox_pred = self.RCNN_bbox_pred(self._head_to_tail(pooled_feat))  # [B*128, 4]
        # classification
        n_roi = pooled_feat.size(0)
        support_mat = []
        query_mat = []
        batch_size = support_feat.size(0)
        #这里querymeat的生成和一般的不一样  这个函数 主要生成辅助的支持集 ，但是这里似乎没有支持集 不知道咋搞的
        for query_feat, target_feat in zip(pooled_feat.chunk(batch_size, dim=0), support_feat.chunk(batch_size, dim=0)):
            # query_feat [128, c, 7, 7], target_feat [1, shot, c, 7, 7]
            query_feat = self.attention(query_feat)
            target_feat = self.attention(target_feat.squeeze(0))
            query_feat = self.attention2(query_feat)
            target_feat = self.attention2(target_feat)
            target_feat = target_feat.unsqueeze(0)
            target_feat = target_feat.view(1, self.n_shot, 1024, -1).transpose(2, 3)  # [1, shot, 49, c]
            target_feat = target_feat.repeat(query_feat.size(0), 1, 1, 1)  # [128, shot, 49, c]
            query_feat = query_feat.view(query_feat.size(0), 1024, -1).transpose(1, 2)  # [128, 49, c]
            if self.pos_encoding:
                target_feat = self.pos_encoding_layer(target_feat.view(-1, 49, 1024)).view(-1, self.n_shot, 49, 1024)
                query_feat = self.pos_encoding_layer(query_feat)
            support_mat += [target_feat]
            query_mat += [query_feat]
        support_mat = torch.cat(support_mat, 0).transpose(0, 1)  # [shot, B*128, 49, c]
        query_mat = torch.cat(query_mat, 0)  # [B*128, 49, c]
        dense_support_feature = []
        q1_matrix = self.rcnn_adapt_q1_layer(query_mat)
        q1_matrix = q1_matrix - q1_matrix.mean(1, keepdim=True)
        q2_matrix = self.rcnn_adapt_q2_layer(query_mat)
        q2_matrix = q2_matrix - q2_matrix.mean(1, keepdim=True)
        q3_matrix = self.rcnn_adapt_q3_layer(query_mat)
        q3_matrix = q3_matrix - q3_matrix.mean(1, keepdim=True)  
        
        for i in range(self.n_shot):
            single_s_mat = support_mat[i]
            k1_matrix = self.rcnn_adapt_k1_layer(single_s_mat)
            k1_matrix = k1_matrix - k1_matrix.mean(1, keepdim=True)
            k2_matrix = self.rcnn_adapt_k2_layer(single_s_mat)
            k2_matrix = k2_matrix - k2_matrix.mean(1, keepdim=True)
            k3_matrix = self.rcnn_adapt_k3_layer(single_s_mat)
            k3_matrix = k3_matrix - k3_matrix.mean(1, keepdim=True)

            support_adaptive_attention_weight1 = torch.bmm(q1_matrix, q2_matrix.transpose(1, 2)) / math.sqrt(self.rcnn_reduce_dim) 
            support_adaptive_attention_weight1 = F.softmax(support_adaptive_attention_weight1, dim=2)

            
            support_adaptive_attention_weight2 = torch.bmm(k1_matrix, k1_matrix.transpose(1, 2)) / math.sqrt(self.rcnn_reduce_dim) 
            support_adaptive_attention_weight2 = F.softmax(support_adaptive_attention_weight2, dim=2)            

            support_adaptive_attention_weight3 = torch.bmm(q3_matrix, k3_matrix.transpose(1, 2)) / math.sqrt(self.rcnn_reduce_dim) 
            support_adaptive_attention_weight3 = F.softmax(support_adaptive_attention_weight3, dim=2)
            
            support_adaptive_attention_feature1 = torch.bmm(support_adaptive_attention_weight1, query_mat)  # [B, hw, 1024]

            unary_term = self.rcnn_unary_layer(support_adaptive_attention_weight2)  # [n_roi, 49, 1]
            unary_term = F.softmax(unary_term, dim=1)
            
            support_adaptive_attention_weight = support_adaptive_attention_weight3 + unary_term.transpose(1, 2)

            support_adaptive_attention_feature2 = torch.bmm(support_adaptive_attention_weight, single_s_mat)  # [B, hw, 1024]
            
            support_adaptive_attention_feature = torch.cat([support_adaptive_attention_feature1,
                                                             support_adaptive_attention_feature2],dim=2)
            
            dense_support_feature += [support_adaptive_attention_feature]
        dense_support_feature = torch.stack(dense_support_feature, 0).mean(0)  # [n_roi, 49, 4096]
        
        if self.attention_type == 'concat':
            correlation_feat = dense_support_feature
        elif self.attention_type == 'product':
            correlation_feat = query_mat * dense_support_feature           
            
        correlation_feat2 = correlation_feat # [n_roi, 49, 128]
        correlation_feat2 = F.normalize(self.avgpool2(correlation_feat2.transpose(1,2).view(-1,2048,7,7)).view(-1,2048))
        f1, f2 = torch.split(correlation_feat2, [1024, 1024], dim=1)
        f1 = self.rcnn_transform_layer2(f1)
        f2 = self.rcnn_transform_layer2(f2)

        # corrlation 就是metaclm模块
        correlation_feat2 = torch.cat([f1,f2],dim=1)
        correlation_feat = self.rcnn_transform_layer(correlation_feat)  # [B*128, 49, rcnn_d]
        cls_score = self.output_score_layer(correlation_feat.view(n_roi, -1))
        cls_prob = F.softmax(cls_score, 1)  # [B*128, 1]

        return bbox_pred, cls_prob, cls_score, correlation_feat2

class FFN(nn.Module):
    def __init__(self, in_channel, hidden, drop_prob=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(in_channel, hidden)
        self.linear2 = nn.Linear(hidden, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model=1024, max_len=49):
        super(PositionalEncoding, self).__init__()
        #print('positionencoding')
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / float(d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = Variable(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        x = x + self.pe.to(x.device)
        return x


class hANMCL(_hANMCL):
    def __init__(self, classes, attention_type, rpn_reduce_dim=256, rcnn_reduce_dim=256, gamma=0.1, 
                num_layers=101, pretrained=False, num_way=2, num_shot=5, pos_encoding=True):
        self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
        self.dout_base_model = 1024
        self.pretrained = pretrained
        _hANMCL.__init__(self, classes, attention_type, rpn_reduce_dim, rcnn_reduce_dim, gamma, 
                                    n_way=num_way, n_shot=num_shot, pos_encoding=pos_encoding)

    def _init_modules(self):
        resnet = resnet101()
        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

        # Build resnet. (base -> top -> head)
        # 这是一个列表 model.train会对 这个堆叠的一部分进行冻结
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
        self.RCNN_top = nn.Sequential(resnet.layer4)  # 1024 -> 2048
        # build rcnn head
        self.RCNN_bbox_pred = nn.Linear(2048, 4)

        # Fix blocks 
        for p in self.RCNN_base[0].parameters(): p.requires_grad=False
        for p in self.RCNN_base[1].parameters(): p.requires_grad=False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            # 可能是有部分是固定的？ 然后只有两层需要 设置为train 训练模式以便进行反向传播和参数更新。
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)  # [128, 2048]
        return fc7
