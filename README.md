# 实验 Readme：基于 OpenCV 的局部特征检测、描述与图像匹配
## 一、实验目的
掌握使用 OpenCV 进行 ORB 特征检测、描述子计算与匹配的方法。
理解 RANSAC 算法剔除错误匹配的原理及其在估计单应矩阵中的作用。
通过透视变换实现目标定位，并比较不同 nfeatures 参数对匹配效果的影响。
分析特征点、描述子、汉明距离等概念，以及 ORB 对旋转、尺度、平移的鲁棒性来源。

## 二、实验环境
操作系统：Linux / Windows / macOS
编程语言：Python 3.x
主要库：OpenCV (4.x)、NumPy、Matplotlib、Tkinter（可选）

## 三、实验数据
box.png：模板图像（目标物体）
box_in_scene.png：场景图像（包含目标物体，存在透视变形）

## 四、实验任务及要求
### 任务1：ORB 特征检测
使用 cv2.ORB_create(nfeatures=1000) 创建检测器。
对两幅图像分别调用 detectAndCompute() 得到关键点和描述子。
用 cv2.drawKeypoints() 可视化关键点。
输出两幅图像的关键点数量以及描述子维度。
提交物：box_keypoints.jpg、scene_keypoints.jpg；关键点数量、描述子维度。

### 任务2：ORB 特征匹配
创建暴力匹配器 cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)。
对模板图和场景图的描述子进行匹配，按距离从小到大排序。
显示前 30 或 50 个匹配结果并保存。
输出总匹配数量。
提交物：orb_matches.jpg（初始匹配图）；总匹配数量。

### 任务3：RANSAC 剔除错误匹配
从匹配结果中提取对应点坐标。
使用 cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 估计单应矩阵，得到内点 mask。
根据 mask 绘制 RANSAC 后的内点匹配图。
输出总匹配数量、内点数量、内点比例 = 内点数 / 总匹配数。
提交物：ransac_matches.jpg；Homography 矩阵；总匹配数、内点数、内点比例。

### 任务4：目标定位
获取 box.png 的四个角点 (0,0), (w,0), (w,h), (0,h)。
使用 cv2.perspectiveTransform() 将角点投影到场景图中。
用 cv2.polylines() 在场景图中绘制绿色边框。
保存结果图并说明定位是否成功。
提交物：target_localization.jpg；简要说明定位成功与否。

### 任务6：参数对比实验（nfeatures 参数）
分别设置 nfeatures = 500, 1000, 2000。
对每组参数重复上述任务1‑4的流程（可仅记录关键指标）。
记录并比较：模板图关键点数、场景图关键点数、匹配数量、RANSAC 内点数、内点比例、定位是否成功。
分析特征点数量对匹配效果和定位精度的影响。
提交物：参数对比表格（含上述指标）及分析结论。

## 五、实验步骤（简述）
图像读取：读入 box.jpg 和 box_in_scene.jpg。
ORB 检测：对每张图像提取关键点和描述子。
暴力匹配：得到原始匹配对，按距离排序。
RANSAC 滤波：计算单应矩阵，获取内点匹配。
目标定位：将模板角点投影到场景中并绘制边框。
参数对比：循环执行步骤 2‑5 对于不同 nfeatures，记录结果。
结果可视化：使用 Matplotlib 生成汇总图 all_results.jpg。

## 六、实验结果示例
关键点数量及描述子维度（nfeatures=1000）
图像	关键点数量	描述子维度
box.png	1000	32 (256 bits)
box_in_scene.png	1000	32 (256 bits)
匹配与 RANSAC 结果（nfeatures=1000）
初始匹配数：例如 456
RANSAC 内点数：例如 212
内点比例：例如 46.5%
定位是否成功：是
参数对比表（示例）
nfeatures	模板关键点	场景关键点	匹配数	内点数	内点比例	定位成功
500	500	500	230	98	42.6%	是
1000	1000	1000	456	212	46.5%	是
2000	2000	2000	789	301	38.1%	是
结论：并非特征点越多越好 → 特征点过多时，错误匹配比例可能升高，内点比例下降；合适的 nfeatures 应在保证足够匹配数的同时维持较高的内点比例。
## 九、输出文件说明
文件名	说明
box_keypoints.jpg	box.png 的关键点可视化
scene_keypoints.jpg	scene.png 的关键点可视化
orb_matches.jpg	初始 ORB 匹配结果（前 50 条）
ransac_matches.jpg	RANSAC 过滤后的内点匹配图
target_localization.jpg	最终定位结果（绿色边框）
all_results.jpg	汇总图（含特征点图、匹配图、定位图及参数对比表格）
## 十、结论
本实验成功实现了基于 ORB 的特征检测与匹配，并通过 RANSAC+Homography 完成了平面目标的透视定位。参数对比表明，合理选择 nfeatures 能在速度与精度之间取得平衡，特征点过多反而可能降低内点比例。ORB 速度快、适合实时应用，但其对剧烈透视变化的抗性有限；若要处理大视角差异，可考虑 SIFT 等更稳定的算法。
