"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  roidb = imdb.roidb #会直接对roidb进行影响
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
  for i in range(len(imdb.image_index)):
    roidb[i]['img_id'] = imdb.image_id_at(i)
    roidb[i]['image'] = imdb.image_path_at(i)
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]    #如果不是coco数组 则通过 PIL获得img的大小 并且设置roidb的图片大小
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    # (num_obj, num_class)
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)  #一个列表 roi对哪个类别更相似
    roidb[i]['max_classes'] = max_classes # 最可能的解
    roidb[i]['max_overlaps'] = max_overlaps# 与哪个gt重叠区域 更大也是个列表函数
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)

      # trim the ratio into 0.5 ~ 2.
      # remark need_crop if the ratio over that range
      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0

      ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    # return sorted ratio list, index
    # ex. [0.5, 0.5, 1., 1.6, 2. 2.]
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True):
  """
  Combine multiple roidbs
  """
  print(imdb_names)

  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()
      print('done')

    print('Preparing training data...')

    prepare_roidb(imdb)
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb
  
  def get_roidb(imdb_name):
    print("666+"+imdb_name)
    imdb = get_imdb(imdb_name) #直接从相对应的路径中获得影响数据集
    #print("666+" + imdb)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    #method为gt  调用了
    #哪里调用了gt_roidb？？ 答：imdb.py中的set_proposal_method函数 之后会执行pascal_voc中的gt_roidb不清楚何时执行

    #进入pascal_voc下的gt_roidb函数，
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    #后续imdb会调用gt_roidb
    #通过对影像数据集进行操作 生成
    roidb = get_training_roidb(imdb)
    return roidb

  #这是主函数 ]我们可以知道对于每个数据集返回了带有该数据集信息的imdb和包含每张影像感兴趣区域信息的roidb（包括每张影像点目标和影像宽高等信息），事实上roidb属于imdb。
  #通过[get_roidb(s) for s in imdb_names.split('+')]  roi影像应该是 标注出来的图片 gt框是真值框 其余
  #voc1来说 s的值为pascal_5_set1  此时列表为roidb 是通过getroidb得到的 如果是 imdb
  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]

  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)

  if training:
    roidb = filter_roidb(roidb)

  ratio_list, ratio_index = rank_roidb_ratio(roidb)
  #对roidb进行 最大宽高比的整理 进行排序后的结果 下文有没有用到不确定
  return imdb, roidb, ratio_list, ratio_index
