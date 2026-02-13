#!/bin/bash

mini_imageNet_url="https://dataset-hub.oss-cn-hangzhou.aliyuncs.com/public-unzip-dataset/tany0699/mini_imagenet100/master/train.zip?Expires=1771013796&OSSAccessKeyId=LTAI5tAoCEDFQFyV5h8unjt8&Signature=7TCCzm5ITwEegO%2BYGn%2F8HqlfbQM%3D&response-content-disposition=attachment%3B"

# 1. 下载train.zip数据
echo "Downloading train.zip..."
mkdir -p datasets/mini_imagenet
curl -L "$mini_imageNet_url" -o datasets/mini_imagenet/train.zip

# 2. 解压到datasets/mini_imagenet下
echo "Unzipping train.zip..."
unzip -o datasets/mini_imagenet/train.zip -d datasets/mini_imagenet/

# 3. 删除压缩包（可选）
# rm datasets/mini_imagenet/train.zip

echo "Done."