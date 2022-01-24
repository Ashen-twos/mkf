# 构建目录
mkdir -p ILSVRC2012  #基本目录
mkdir -p ILSVRC2012/raw-data  
mkdir -p ILSVRC2012/raw-data/imagenet-data 
mkdir -p ILSVRC2012/raw-data/imagenet-data/train/ #训练集
mkdir -p ILSVRC2012/raw-data/imagenet-data/validation/  #验证集
mkdir -p ILSVRC2012/raw-data/imagenet-data/bounding_boxes #bbox数据集

# 做验证集(解压时间久)
# 将ILSVRC2012_img_val.tar解压至ILSVRC2012/raw-data/imagenet-data/validation/目录下
tar -xvf ILSVRC2012_img_val.tar -C ILSVRC2012/raw-data/imagenet-data/validation/
# 执行验证集处理脚本
python preprocess_imagenet_validation_data.py ILSVRC2012/raw-data/imagenet-data/validation/ imagenet_2012_validation_synset_labels.txt

# 做bounding box数据(解压时间超级久)
# 将ILSVRC2012_bbox_train_v2.tar解压至ILSVRC2012/raw-data/imagenet-data/bounding_boxes目录下
tar -xvf ILSVRC2012_bbox_train_v2.tar.gz -C ILSVRC2012/raw-data/imagenet-data/bounding_boxes/
# 执行bounding box处理脚本
python process_bounding_boxes.py ILSVRC2012/raw-data/imagenet-data/bounding_boxes/ imagenet_lsvrc_2015_synsets.txt | sort > ILSVRC2012/raw-data/imagenet_2012_bounding_boxes.csv

# 做训练集(解压时间十分非常之久)
# 将ILSVRC2012_img_train.tar解压至ILSVRC2012/raw-data/imagenet-data/train/目录下
mv ILSVRC2012_img_train.tar ILSVRC2012/raw-data/imagenet-data/train/ && cd ILSVRC2012/raw-data/imagenet-data/train/  
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAE ; do mkdir -p "${NAE%.tar}"; tar -xvf "${NAE}" -C "${NAE%.tar}"; rm -f "${NAE}"; done

# 执行构建数据集的主程序
python build_imagenet_data.py --train_directory=ILSVRC2012/raw-data/imagenet-data/train --validation_directory=ILSVRC2012/raw-data/imagenet-data/validation --output_directory=ILSVRC2012/ --imagenet_metadata_file=imagenet_metadata.txt --labels_file=imagenet_lsvrc_2015_synsets.txt --bounding_box_file=ILSVRC2012/raw-data/imagenet_2012_bounding_boxes.csv