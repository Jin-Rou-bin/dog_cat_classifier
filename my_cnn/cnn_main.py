import os
os.environ['TORCH_HOME'] = 'torch_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader, Subset


# ===== 图像预处理流水线 =====
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ===== 划分训练/验证集 =====
print("👉【1】扫描 all_data 获取全部文件列表...")
temp_ds = datasets.ImageFolder(root="all_data")
total = len(temp_ds)
print(f"✅【1】一共找到了 {total} 张图片")

train_len = int(0.8 * total)
val_len = total - train_len
print(f"👉【2】划分：训练{train_len}张，验证{val_len}张")

train_idx, val_idx = random_split(
    range(total),
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42)
)

print("👉【3】给训练集绑定训练增强transform...")
full_train_ds = datasets.ImageFolder("all_data", transform=train_tf)

print("👉【4】给验证集绑定验证transform...")
full_val_ds   = datasets.ImageFolder("all_data", transform=val_tf)

print("👉【5】用索引切分数据集（逻辑分割，不复制文件）...")
train_ds = Subset(full_train_ds, train_idx.indices)
val_ds   = Subset(full_val_ds,   val_idx.indices)

print("👉【6】创建train_loader，num_workers=0（Windows防卡死）...")
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

print("👉【7】创建val_loader...")
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

print("✅数据集全部就绪，准备搭模型！")


# ===== 搭建模型、加载预训练权重、冻结 =====
print("👉【8】设置device（cuda/ cpu）...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅使用设备：{device}")

print("👉【9】创建resnet18空骨架，加载本地预训练权重...")
model = models.resnet18(weights=None)
model.load_state_dict(torch.load("torch_cache/hub/checkpoints/resnet18-f37072fd.pth", map_location="cpu"))

# 冻结前面所有层：不让前面层更新，只学最后一层（迁移学习/微调）
for p in model.parameters():
    p.requires_grad = False

# 替换最后一层为2分类
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = model.to(device)
print("✅模型搭建完成！")

criterion = torch.nn.CrossEntropyLoss()   # 交叉熵损失：分类任务标准损失函数
opt = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
EPOCHS = 6


# ===== 主训练循环 =====
print("\n🚀开始训练循环！")
for ep in range(EPOCHS):
    model.train()
    sum_loss = 0.0
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        opt.zero_grad()      # 清空上一轮梯度
        out = model(imgs)     # 前向传播：算预测值
        loss = criterion(out, labs) # 算预测和真实标签差多少（损失）
        loss.backward()       # 反向传播：算每个参数该怎么调
        opt.step()            # 真的去更新参数
        sum_loss += loss.item()

    model.eval()
    correct = 0
    total_num = 0
    with torch.no_grad(): # 验证不开梯度，省内存、提速
        for imgs, labs in val_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            _, pred = torch.max(out, dim=1)
            total_num += labs.size(0)
            correct += (pred == labs).sum().item()

    acc = 100.0 * correct / total_num
    print(f"📌轮次{ep+1:2d} | loss:{sum_loss/len(train_loader):.3f} | val准确率:{acc:.2f}%")


# ==========保存权重==========
save_path = "my_catdog_best.pth"
torch.save(model.state_dict(), save_path)
print(f"\n✅训练结束！模型已经保存到：{save_path}")
print("👉以后不用再训练，可以直接加载这个文件来预测图片！")