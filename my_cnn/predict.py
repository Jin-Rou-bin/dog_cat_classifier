import os
os.environ['TORCH_HOME'] = 'torch_cache'

import torch
from torchvision import transforms, models
from PIL import Image   # 需要 pillow: pip install pillow

# ---------- 和训练时保持一模一样的设置！非常重要 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理必须和val_tf完全一样！不然分布不对结果乱飘
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 1. 先搭空骨架（和训练时结构必须完全一致！）
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# 2. 把我们刚才存好的、训完的权重装进去
model.load_state_dict(torch.load("my_catdog_best.pth", map_location=device))
model = model.to(device)
model.eval()   # 一定要切评估模式！

# 类别对应关系：ImageFolder自动分配：0=cat，1=dog
classes = ["猫", "狗"]

# ----------------在这里改你要预测的图片路径----------------
img_path = "C:\\Users\\Jin\\Desktop\\test4.png"   # ←把这个换成你的图片名，比如 "my_dog.jpg"
# ---------------------------------------------------------

# 加载图片、做变换
img = Image.open(img_path).convert("RGB")
tensor = transform(img).unsqueeze(0)  # unsqueeze(0): 加一个batch维度 → [1,3,224,224]

# 预测（不开梯度）
with torch.no_grad():
    out = model(tensor.to(device))
    _, idx = torch.max(out, dim=1)

print(f"✅预测结果：这张图是 → {classes[idx.item()]}")