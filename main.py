# ==============================================
# 项目：鸢尾花分类小实验
# 作用：用3种机器学习算法，根据花的4个尺寸判断花的品种
# ==============================================

# 从sklearn工具包里拿出我们需要的功能
# load_iris：自带的鸢尾花数据集（不用自己找文件）
from sklearn.datasets import load_iris

# train_test_split：把数据分成「训练用」和「测试用」两堆
from sklearn.model_selection import train_test_split

# KNeighborsClassifier：K近邻算法，一种简单的分类模型
from sklearn.neighbors import KNeighborsClassifier

# RandomForestClassifier：随机森林算法
from sklearn.ensemble import RandomForestClassifier

# DecisionTreeClassifier：决策树算法
from sklearn.tree import DecisionTreeClassifier

# accuracy_score：用来算“模型猜得准不准”（准确率）
from sklearn.metrics import accuracy_score

# matplotlib.pyplot：画图工具，别名plt，方便写
import matplotlib.pyplot as plt

# --------------------------
# 下面两行：解决中文变成小方框的bug（Windows才这么写）
# --------------------------
# 设置画图用的字体为SimHei（黑体，系统自带，能显示汉字）
plt.rcParams["font.family"] = ["SimHei"]
# 让图里的负号“−”不要变成方框
plt.rcParams["axes.unicode_minus"] = False

# --------------------------
# 第1步：加载鸢尾花数据
# --------------------------
# 把鸢尾花数据集拿出来放到变量iris里
iris = load_iris()

# X = 特征：每一朵花的4个测量数据（花萼长、花萼宽、花瓣长、花瓣宽）
# X就是输入：给模型看什么
X = iris.data

# y = 标签：每一朵花对应的品种编号：0、1、2，共3种花
# y就是答案：模型要猜什么
y = iris.target

# feature_names：4个特征对应的中文/英文名字（"sepal length"…）
feature_names = iris.feature_names

# class_names：3种花真正的英文名字
class_names = iris.target_names

# --------------------------
# 第2步：切分数据 → 训练集 + 测试集
# --------------------------
# test_size=0.3：拿30%的数据当“考试题（测试集）”
# 剩下70%当“课本（训练集）”，给模型学习规律
# random_state=42：固定随机方式，每次跑结果都一样，方便对比
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# --------------------------
# 第3步：准备3个不同的AI模型
# --------------------------
# 建一个字典：名字 → 对应的模型对象
models = {
    # n_neighbors=3：KNN找离它最近的3个邻居来投票
    "K近邻(KNN)": KNeighborsClassifier(n_neighbors=3),
    # n_estimators=50：随机森林用50棵小树一起做决定
    "随机森林": RandomForestClassifier(n_estimators=50, random_state=42),
    # random_state=42：固定随机，每次结果不变
    "决策树": DecisionTreeClassifier(random_state=42)
}

# --------------------------
# 第4步：挨个训练模型、算准确率
# --------------------------
# 打印一行等号，把界面弄好看
print("=" * 35)
# 打印标题
print("        模型准确率对比")
print("=" * 35)

# 循环：把字典里每一个模型拿出来跑一遍
# name = 模型名字；model = 那个模型本身
for name, model in models.items():
    # fit = “训练/学习”：把训练集喂给模型，让它找规律
    model.fit(X_train, y_train)

    # predict = “预测/做题”：给它没见过的测试数据，让它猜答案
    y_pred = model.predict(X_test)

    # accuracy_score：拿“模型猜的答案y_pred”和“真实答案y_test”对比，算出准确率0~1
    acc = accuracy_score(y_test, y_pred)

    # 打印结果：:.2% → 把0.9666…变成96.67%，人看着舒服
    print(f"{name:10s}  准确率：{acc:.2%}")

# --------------------------
# 第5步：画第一张图：特征重要性（只有随机森林能干这个！）
# --------------------------
# 从字典取出随机森林这个模型
rf = models["随机森林"]

# figure：新建一张画布，figsize设置宽和高（英寸）
plt.figure(figsize=(8, 5))

# barh = horizontal bar：横向柱状图
# x轴：重要性数值；y轴：4个特征名字
# rf.feature_importances_：随机森林算出来“每个特征有多重要”
plt.barh(feature_names, rf.feature_importances_)

# xlabel：给X轴写标签
plt.xlabel("特征重要性")

# title：给整张图写标题
plt.title("各特征对鸢尾花分类的贡献程度")

# tight_layout：自动调整边距，防止字被切掉
plt.tight_layout()

# show：把这张图画出来！程序会在这里暂停，等你关窗口才往下走
plt.show()

# --------------------------
# 第6步：画第二张图：散点图（花瓣长 vs 花瓣宽）
# 注意：必须关掉上面那张图，这段才会执行！
# --------------------------
# 再新建另一张画布
plt.figure(figsize=(8, 5))

# 循环i=0,1,2：对应3种花；给每一种花一个颜色
# enumerate：同时拿到序号i和颜色c
for i, c in enumerate(["red", "green", "blue"]):
    # mask = y == i：布尔数组 → 挑出“正好是第i种花”的所有样本
    mask = y == i

    # scatter = 散点：画一堆点
    # X[mask,2]：只取这批样本的第2列（花瓣长度）做x坐标
    # X[mask,3]：只取这批样本的第3列（花瓣宽度）做y坐标
    # c=c：点的颜色；label=名字，给图例用
    plt.scatter(X[mask, 2], X[mask, 3], c=c, label=class_names[i])

# X轴说明：花瓣长度
plt.xlabel("花瓣长度 (cm)")
# Y轴说明：花瓣宽度
plt.ylabel("花瓣宽度 (cm)")
# 图标题
plt.title("三种鸢尾花：花瓣长—花瓣宽分布")
# legend：显示图例（红=啥、绿=啥、蓝=啥）
plt.legend()
# 自动调边距
plt.tight_layout()
# 显示这张散点图
plt.show()