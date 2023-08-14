import json
def duplicate_image():

    # 读取第一个 JSON 文件
    with open('../../data/pascal/annotations/pascal15_train/novel1 (7).json', 'r') as f1:
        data1 = json.load(f1)

    # 提取第一个 JSON 文件中的图片唯一标识符到列表
    image_ids1 = [image['id'] for image in data1['images']]

    # 读取第二个 JSON 文件
    with open('../../data/pascal/annotations/pascal15_train/train_split1_hANCML.json', 'r') as f2:
        data2 = json.load(f2)

    # 检查第二个 JSON 文件中的图片是否重复出现
    duplicated_images = []
    for image in data2['images']:
        if image['id'] in image_ids1:
            duplicated_images.append(image)

    # 打印重复的图片信息
    if duplicated_images:
        print("以下图片在两个 JSON 文件中重复出现：")
        for image in duplicated_images:
            print(f"图片ID: {image['id']}, 文件名: {image['file_name']}")
    else:
        print("两个 JSON 文件中没有重复出现的图片。")
def len_annotations():
    # 读取COCO格式的JSON文件
    with open('../../data/pascal/annotations/pascal15_train/novel1 (7).json', 'r') as f:
        data = json.load(f)

    # 统计不同类别的注释数据
    categories = {}
    annotations = data['annotations']
    for annotation in annotations:
        category_id = annotation['category_id']
        if category_id not in categories:
            categories[category_id] = 1
        else:
            categories[category_id] += 1

    # 打印每个类别的注释数据数量
    print(len(categories))
    for category_id, count in categories.items():
        category_name = [category['name'] for category in data['categories'] if category['id'] == category_id][0]
        print(f"类别 '{category_name}' 具有 {count} 条注释数据。")
def img_ann_lack():#判断是否img
    # 读取 JSON 文件
    with open('../../data/pascal/annotations/pascal15_train/train_split1_hANCML.json', 'r') as f:
        data = json.load(f)

    # 获取图像和注释列表
    images = data['images']
    annotations = data['annotations']

    # 检查存在注释但没有图像的情况
    missing_images_annotations = []
    annotation_image_ids = set(annotation['image_id'] for annotation in annotations)
    for image in images:
        if image['id'] not in annotation_image_ids:
            missing_images_annotations.append(image)

    # 检查存在图像但没有注释的情况
    missing_annotations_images = []
    image_ids = set(image['id'] for image in images)
    for annotation in annotations:
        if annotation['image_id'] not in image_ids:
            missing_annotations_images.append(annotation)

    # 打印结果
    if missing_images_annotations:
        print(f"{len(missing_images_annotations)} 条注释存在但没有相应的图像：")
        for image in missing_images_annotations:
            print(f"图像 ID: {image['id']}, 文件名: {image['file_name']}")

    if missing_annotations_images:
        print(f"{len(missing_annotations_images)} 条图像存在但没有相应的注释：")
        for annotation in missing_annotations_images:
            print(f"注释 ID: {annotation['id']}, 图像 ID: {annotation['image_id']}")
#duplicate_image()
#img_ann_lack()
len_annotations()