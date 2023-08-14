
import json
import os
def get_category_id(category_name):
    categories = {
        "aeroplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "person": 15,
        "pottedplant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tvmonitor": 20
    }
    return categories.get(category_name, None)


PASCAL_VOC_BASE_CATEGORIES = {
    1: ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor'],
    2: ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor'],
    3: ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor'],
}


# 读取 Coco JSON 文件

with open('pascal_trainval0712.json', 'r') as f:
    data = json.load(f)

# 获取图像和注释列表
images = data['images']
annotations = data['annotations']

for file_name, keep_categories in PASCAL_VOC_BASE_CATEGORIES.items():
    # 构建需要保留的类别 ID 集合
    keep_ids = set()
    for i in keep_categories:
        keep_ids.add(get_category_id(i))
    keep_category_ids = set()
    for annotation in annotations:
        if annotation['category_id'] in keep_ids:
            keep_category_ids.add(annotation['image_id'])
    # print(len(keep_ids))
    print(len(keep_category_ids))
    print(len(images))
    # 筛选出保留的图像和注释
    filtered_images = []
    filtered_annotations = []
    filtered_categories = []
    for image in images:
        if image['id'] in keep_category_ids:
            filtered_images.append(image)

    for annotation in annotations:
        if annotation['category_id'] in keep_ids:
            filtered_annotations.append(annotation)
    print(len(filtered_annotations))
    print(len(annotations))
    for cat in data['categories']:
        if cat['name'] in keep_categories:
            filtered_categories.append(cat)
    # 将处理后的结果保存到新的 JSON 文件中
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }
    output_file = f"voc_base{file_name}.json"
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f)

    print(f"File '{output_file}' generated.")