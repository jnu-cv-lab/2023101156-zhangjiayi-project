import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
import os
from tkinter import Tk, filedialog

font_path = "SimHei.ttf"
# 添加字体
fm.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def select_points_matplotlib(image):
    """使用 matplotlib 选择4个点"""
    points = []
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title("请按顺序点击4个角点（左上、右上、左下、右下），点击完成后关闭窗口")
    ax.axis('off')
    
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((int(event.xdata), int(event.ydata)))
            ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
            ax.text(event.xdata+5, event.ydata-5, str(len(points)), 
                   fontsize=12, color='red', weight='bold')
            fig.canvas.draw()
            print(f"已选择点 {len(points)}: ({int(event.xdata)}, {int(event.ydata)})")
            if len(points) == 4:
                print("已选择4个点，请关闭窗口")
                plt.close()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    if len(points) == 4:
        return np.float32(points)
    else:
        print(f"选择了 {len(points)} 个点，需要4个点")
        return None

# 创建测试图 (500x500)
img = np.ones((500, 500, 3), dtype=np.uint8) * 255

# 绘制矩形
cv2.rectangle(img, (100, 100), (250, 200), (0, 0, 255), 2)

# 绘制圆
cv2.circle(img, (375, 150), 50, (0, 255, 0), 2)

# 绘制平行线 (水平)
cv2.line(img, (50, 300), (450, 300), (255, 0, 0), 2)
cv2.line(img, (50, 350), (450, 350), (255, 0, 0), 2)

# 绘制垂直线 (与水平线垂直)
cv2.line(img, (300, 250), (300, 450), (0, 255, 255), 2)

plt.figure(figsize=(12, 10))

# 原图
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("原图")
plt.axis('off')

# 1. 相似变换 (旋转+缩放)
center = (250, 250)
M_similar = cv2.getRotationMatrix2D(center, 30, 0.8)
img_similar = cv2.warpAffine(img, M_similar, (500, 500))

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(img_similar, cv2.COLOR_BGR2RGB))
plt.title("相似变换")
plt.axis('off')

# 2. 仿射变换
pts1 = np.float32([[100, 100], [250, 100], [100, 200]])
pts2 = np.float32([[80, 120], [260, 90], [120, 210]])
M_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M_affine, (500, 500))

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(img_affine, cv2.COLOR_BGR2RGB))
plt.title("仿射变换")
plt.axis('off')

# 3. 透视变换（保持原始效果，使用固定输出尺寸）
pts1 = np.float32([[100, 100], [250, 100], [100, 200], [250, 200]])
pts2 = np.float32([[60, 120], [270, 80], [90, 220], [280, 210]])
M_persp = cv2.getPerspectiveTransform(pts1, pts2)
img_persp = cv2.warpPerspective(img, M_persp, (500, 500))

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(img_persp, cv2.COLOR_BGR2RGB))
plt.title("透视变换")
plt.axis('off')

# 结果表格
results = [
    ["相似变换", "保持直线", "保持平行", "保持垂直", "保持圆"],
    ["仿射变换", "保持直线", "保持平行", "不保持垂直", "变为椭圆"],
    ["透视变换", "保持直线", "不保持平行", "不保持垂直", "变为椭圆或复杂曲线"]
]

plt.subplot(2, 3, 5)
plt.axis('tight')
plt.axis('off')
table = plt.table(cellText=results, colLabels=["变换类型", "直线", "平行线", "垂直线", "圆"],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

plt.tight_layout()
plt.savefig('transform_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ 透视畸变校正 ============
# 弹窗选择图片
root = Tk()
root.withdraw()
print("请选择要校正的图片...")
file_path = filedialog.askopenfilename(
    title="选择透视畸变图片",
    filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
)
root.destroy()

def create_distorted_image():
    """创建模拟透视畸变的图像"""
    distorted = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.putText(distorted, "Hello World", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(distorted, "A4 Paper", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(distorted, (80, 300), (420, 420), (0, 0, 0), 2)
    for i in range(4):
        cv2.line(distorted, (80, 300 + i*30), (420, 300 + i*30), (0, 0, 0), 1)
    
    pts_orig = np.float32([[80, 300], [420, 300], [80, 420], [420, 420]])
    pts_dist = np.float32([[60, 310], [440, 290], [100, 430], [430, 440]])
    M_dist = cv2.getPerspectiveTransform(pts_orig, pts_dist)
    return cv2.warpPerspective(distorted, M_dist, (500, 500))

if file_path:
    # 读取用户选择的图片
    distorted_img = cv2.imread(file_path)
    if distorted_img is None:
        print("图片读取失败，使用默认模拟图片")
        distorted_img = create_distorted_image()
else:
    print("未选择文件，使用默认模拟图片")
    distorted_img = create_distorted_image()

# 使用 matplotlib 选择4个角点
print("\n请在弹出的窗口中按顺序点击4个角点：左上、右上、左下、右下")
src_pts = select_points_matplotlib(distorted_img)

if src_pts is not None:
    h, w = distorted_img.shape[:2]
    dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # 计算透视变换矩阵并校正（使用原始图像尺寸）
    M_correct = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(distorted_img, M_correct, (w, h))
    
    # 显示结果
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB))
    plt.title("原始畸变图像")
    plt.axis('off')
    
    # 在原始图像上标记选择的点
    img_with_points = distorted_img.copy()
    for i, pt in enumerate(src_pts):
        cv2.circle(img_with_points, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(img_with_points, str(i+1), tuple(pt.astype(int) + np.array([5, -5])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img_with_points, cv2.COLOR_BGR2RGB))
    plt.title("选择的4个角点")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title("校正后图像")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('correction_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n校正成功！原始尺寸: ({w}, {h})")
else:
    print("未选择足够的点，跳过校正")

print("\n=== 总结 ===")
print("1. 直线保持性：三种变换都保持直线")
print("2. 平行保持性：相似、仿射保持平行；透视不保持")
print("3. 垂直保持性：仅相似保持垂直")
print("4. 圆保持性：仅相似保持圆；仿射和透视使圆变为椭圆")