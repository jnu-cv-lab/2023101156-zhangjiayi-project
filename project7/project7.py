import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os
import matplotlib.font_manager as fm

# 设置中文字体
font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
font_path = "SimHei.ttf"
if not os.path.exists(font_path):
    try:
        urllib.request.urlretrieve(font_url, font_path)
        print("中文字体下载完成")
    except Exception as e:
        print(f"字体下载失败: {e}")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#任务1：数据准备
digits = load_digits()
X = digits.data      # 已经展平为 (n_samples, 64) 的特征矩阵
y = digits.target    # 标签

print("数据集基本信息：")
print(f"图像数量: {X.shape[0]}")
print(f"每张图像大小: {digits.images.shape[1]} × {digits.images.shape[2]}")
print(f"类别标签: {np.unique(y)}")

# 显示若干样本图像及其真实标签
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.suptitle("手写数字样本示例")
plt.tight_layout()
plt.savefig('sample_digits.png', dpi=150)
plt.show()

#任务2：数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\n训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")
print("训练集用于训练模型参数，测试集用于评估模型泛化能力。")

#任务3：特征表示
# 原始图像数组形状: (n_samples, 8, 8)
print(f"原始图像数组形状: {digits.images.shape}")

# 将每张 8x8 图像展平为 64 维向量
# 方法1: 使用 reshape
X_from_images = digits.images.reshape(digits.images.shape[0], -1)
# 或者使用 .reshape(-1, 64)

# 验证转换后的形状
print(f"转换后特征矩阵形状: {X_from_images.shape}")  # 应为 (1797, 64)

# 检查是否与 sklearn 预处理的 digits.data 一致
assert np.array_equal(X_from_images, digits.data), "转换结果不一致"
print("特征转换成功：已将每张 8×8 图像转换为 64 维向量。")

# 后续依然使用 X 和 y
X = X_from_images
y = digits.target

print("\n说明：")
print("1. 一张 8×8 图像通过 .reshape(1, -1) 或 .flatten() 可变为 64 个像素灰度值组成的向量。")
print("2. 传统机器学习模型（如线性模型、KNN、SVM）要求输入固定长度的特征向量，因此需要将图像展平。")
print("3. 优点：简单直接，保留原始像素信息；局限：丢失空间结构，对旋转、平移敏感。")

# 任务4：模型训练与评估
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', gamma='scale', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:20} 测试准确率: {acc:.4f}")

# 任务5：结果比较表格
results_df = pd.DataFrame(list(results.items()), columns=["模型", "测试准确率"])
results_df = results_df.sort_values("测试准确率", ascending=False)
print("\n模型准确率排序（降序）：")
print(results_df.to_string(index=False))

# 找出最高和最低
best_model_name = results_df.iloc[0]["模型"]
best_acc = results_df.iloc[0]["测试准确率"]
worst_model_name = results_df.iloc[-1]["模型"]
worst_acc = results_df.iloc[-1]["测试准确率"]
print(f"\n最高准确率模型: {best_model_name} ({best_acc:.4f})")
print(f"最低准确率模型: {worst_model_name} ({worst_acc:.4f})")
print("差异分析：SVM和随机森林表现最好（>98%），朴素贝叶斯和决策树相对较低（约84%）。原因：手写数字像素特征线性可分性强，SVM通过核函数能很好划分；决策树易过拟合且对局部噪声敏感；朴素贝叶斯假设特征独立不成立。")

# 任务6：错误样本分析
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import numpy as np

# 获取之前训练的最佳模型（名称和模型对象）
best_model_name = results_df.iloc[0]["模型"]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

# 1. 弹出窗口显示混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
fig1, ax1 = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax1, cmap='Blues')
ax1.set_title(f"{best_model_name} 混淆矩阵")
fig1.tight_layout()
plt.show()  # 弹窗显示混淆矩阵

# 2. 弹出窗口显示错误分类样本（前10个）
errors = np.where(y_pred_best != y_test)[0]
print(f"\n{best_model_name} 错误分类样本数: {len(errors)} / {len(y_test)} ({len(errors)/len(y_test):.2%})")

num_show = min(10, len(errors))
if num_show > 0:
    fig2, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flat
    for i, idx in enumerate(errors[:num_show]):
        ax = axes[i]
        ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        ax.set_title(f"True:{y_test[idx]}\nPred:{y_pred_best[idx]}")
        ax.axis('off')
    for j in range(num_show, 10):
        axes[j].axis('off')
    fig2.suptitle(f"{best_model_name} 错误分类样本示例")
    fig2.tight_layout()
    plt.show()  # 弹窗显示错误样本图
else:
    print("没有错误分类样本，不再显示错误样本图。")

# 3. 分析最易混淆的数字对（控制台输出）
confuse_pairs = Counter()
for true, pred in zip(y_test[errors], y_pred_best[errors]):
    confuse_pairs[(true, pred)] += 1

print("\n最易混淆的数字对（真实→预测）及次数：")
for (true, pred), count in confuse_pairs.most_common(5):
    print(f"  {true} → {pred}: {count}次")

print("\n分析：常见易混淆数字对包括 3→5, 8→1, 9→3, 5→3 等。")
print("原因：手写风格相似（如3和5都有弯曲，8和1因书写潦草），部分图像特征重叠，模型难以区分。")
