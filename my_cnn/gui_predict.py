import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES

# ====================== 配置区 ======================
MODEL_PATH = "my_catdog_best.pth"
CLASS_NAMES = ["猫", "狗"]   # 0=猫，1=狗，和训练时顺序严格对应
IMG_SIZE = 224

# ⭐预处理必须和训练完全一致
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====================== 加载模型 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    messagebox.showwarning("警告", f"找不到模型文件：{MODEL_PATH}\n确认和gui_predict.py在同一个文件夹！")

model.to(device)
model.eval()   # 评估模式，关闭dropout

# ====================== 单张预测函数 ======================
def predict_image(img_path):
    """
    输入：图片路径
    返回：标签名、自信度、图片对象、猫概率(p0)、狗概率(p1)
    p0 → 类别0 = 猫
    p1 → 类别1 = 狗
    """
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():   # 关闭梯度，推理不需要反向传播
        out = model(tensor)
        prob = F.softmax(out, dim=1)  # 转成0~1的概率，两者和≈1
        p0 = prob[0, 0].item()
        p1 = prob[0, 1].item()
        _, pred_idx = torch.max(out, dim=1)

    label = CLASS_NAMES[pred_idx.item()]
    confidence = prob[0, pred_idx.item()].item()  # 最大的那个概率就是自信度
    return label, confidence, img, p0, p1

# ====================== GUI窗口逻辑 ======================
class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("猫狗分类预测工具")
        self.root.geometry("920x680")

        self.current_index = 0
        self.image_list = []

        # 开启拖拽（tkinterdnd2功能）
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

        # 顶部按钮栏
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Button(top_frame, text="📂选择文件夹", command=self.select_folder, width=14).pack(side=tk.LEFT)
        tk.Button(top_frame, text="⬅上一张", command=self.prev_image, width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(top_frame, text="➡下一张", command=self.next_image, width=10).pack(side=tk.LEFT, padx=4)

        self.folder_label = tk.Label(top_frame, text="提示：可以把图片/文件夹拖进窗口！", anchor="w")
        self.folder_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # 图片显示区
        self.img_label = tk.Label(root, text="把图片拖进来 或者 点「选择文件夹」", bg="#eeeeee")
        self.img_label.pack(pady=12)

        # 结果文字区
        self.result_label = tk.Label(root, text="", font=("Microsoft YaHei",15), justify="left")
        self.result_label.pack(pady=6)

        self.progress_label = tk.Label(root, text="", font=("Microsoft YaHei",11))
        self.progress_label.pack()

        # 文件列表
        self.listbox = tk.Listbox(root, height=11)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

    def on_drop(self, event):
        """处理拖入文件/文件夹"""
        path = event.data.strip("{}")
        if os.path.isdir(path):
            self.load_images_from_folder(path)
        elif os.path.isfile(path):
            self.image_list = [path]
            self.current_index = 0
            self.listbox.delete(0, tk.END)
            self.listbox.insert(0, os.path.basename(path))
            self.show_image()

    def load_images_from_folder(self, folder):
        """加载整个文件夹里的图片"""
        self.folder_label.config(text=f"当前文件夹：{folder}")
        files = os.listdir(folder)
        image_files = [f for f in files if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        image_files.sort()
        if not image_files:
            messagebox.showinfo("提示","这个文件夹里没找到图片！")
            return
        self.image_list = [os.path.join(folder,f) for f in image_files]
        self.current_index = 0
        self.listbox.delete(0, tk.END)
        for f in image_files:
            self.listbox.insert(tk.END,f)
        self.show_image()

    def select_folder(self):
        folder = filedialog.askdirectory(title="选择放图片的文件夹")
        if folder:
            self.load_images_from_folder(folder)

    def show_image(self):
        if not self.image_list:
            return
        img_path = self.image_list[self.current_index]
        fname = os.path.basename(img_path)
        try:
            label, conf, img, p0, p1 = predict_image(img_path)
            img.thumbnail((550,420))
            photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=photo, text="")
            self.img_label.image = photo
            # p0永远=猫，p1永远=狗，不会写反
            self.result_label.config(
                text=f"文件名：{fname}\n预测：{label}｜自信度：{conf*100:.2f}%\n"
                     f"猫概率：{p0*100:.2f}%  狗概率：{p1*100:.2f}%"
            )
            self.progress_label.config(text=f"{self.current_index+1} / {len(self.image_list)}")
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.current_index)
        except Exception as e:
            self.result_label.config(text=f"预测失败：{fname}\n错误：{str(e)}")

    def prev_image(self):
        if self.image_list:
            self.current_index = (self.current_index - 1) % len(self.image_list)
            self.show_image()

    def next_image(self):
        if self.image_list:
            self.current_index = (self.current_index + 1) % len(self.image_list)
            self.show_image()

    def on_list_select(self, evt):
        s = self.listbox.curselection()
        if s:
            self.current_index = s[0]
            self.show_image()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = PredictApp(root)
    root.mainloop()