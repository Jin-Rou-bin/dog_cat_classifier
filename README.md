# dog_cat_classifier
基于PyTorch + ResNet18 迁移学习实现猫狗二分类图像识别

## 项目功能
1. 对输入图片判断是猫还是狗
2. 使用迁移学习，在预训练ResNet18基础上微调训练
3. 自动划分训练集/验证集，做数据增强
4. 提供两种预测方式：控制台输出结果、可视化窗口展示图片与预测标签

## 环境
- Python >=3.8
- PyTorch >=2.1.0
- torchvision >=0.16.0
- matplotlib（可视化预测脚本依赖）

安装依赖命令：pip install torch torchvision matplotlib

## 使用方法
1. 准备猫狗图片，放到my_cnn/all_data文件夹，分cat、dog两个子文件夹
2. 运行 cnn_main.py 开始训练
3. 训练结束自动保存模型权重，加载权重做图片预测
   - 控制台预测：运行 my_cnn/predict.py
   - 可视化预测：运行 my_cnn/gui_predict.py

## 说明
数据集图片、训练出来的模型权重文件不会上传github（由.gitignore忽略）
