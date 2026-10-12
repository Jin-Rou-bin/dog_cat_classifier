# dog_cat_classifier
基于PyTorch + ResNet18 迁移学习实现猫狗二分类图像识别

## 项目功能
1. 对输入图片判断是猫还是狗
2. 使用迁移学习，在预训练ResNet18基础上微调训练
3. 自动划分训练集/验证集，做数据增强

## 环境
- Python
- PyTorch
- torchvision

## 使用方法
1. 准备猫狗图片，放到my_cnn/all_data 文件夹，分cat、dog两个子文件夹
2. 运行 main.py 开始训练
3. 训练结束自动保存模型权重，可加载权重做图片预测

## 说明
数据集图片、训练出来的模型权重文件不会上传github（由.gitignore忽略）
