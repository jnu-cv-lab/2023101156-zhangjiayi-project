import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import urllib.request

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

def orb_feature_detection(img, nfeatures=1000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des, orb

def feature_matching(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def ransac_filter(matches, kp1, kp2, reproj_thresh=5.0):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if mask is not None:
        mask = mask.ravel()
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
    else:
        mask = np.zeros(len(matches), dtype=np.uint8)
        inliers = []
    return H, mask, inliers, src_pts, dst_pts

def draw_matches(img1, kp1, img2, kp2, matches, save_path):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(save_path, img_matches)
    return img_matches

def draw_keypoints(img, kp, save_path):
    img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),
                                flags=cv2.DrawMatchesFlags_DEFAULT)
    cv2.imwrite(save_path, img_kp)
    return img_kp

def localize_object(img_scene, H, img_box_shape, save_path):
    h_box, w_box = img_box_shape[:2]
    box_corners = np.float32([[0, 0], [w_box, 0], [w_box, h_box], [0, h_box]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(box_corners, H)
    img_result = img_scene.copy()
    img_result = cv2.polylines(img_result, [np.int32(scene_corners)], True, (0, 255, 0), 3)
    cv2.imwrite(save_path, img_result)
    return img_result, scene_corners

def run_pipeline(img_box, img_scene, nfeatures, save_suffix=""):
    """使用指定nfeatures运行完整流程，返回关键结果和图像"""
    kp_box, des_box, _ = orb_feature_detection(img_box, nfeatures)
    kp_scene, des_scene, _ = orb_feature_detection(img_scene, nfeatures)
    
    matches = feature_matching(des_box, des_scene)
    total_matches = len(matches)
    H, mask, inlier_matches, _, _ = ransac_filter(matches, kp_box, kp_scene)
    inliers = len(inlier_matches)
    inlier_ratio = inliers / total_matches if total_matches > 0 else 0
    
    # 保存关键点图
    img_box_kp = draw_keypoints(img_box, kp_box, f'box_keypoints{save_suffix}.jpg')
    img_scene_kp = draw_keypoints(img_scene, kp_scene, f'scene_keypoints{save_suffix}.jpg')
    
    # 前50匹配图
    num_display = min(50, total_matches)
    matches_top50 = matches[:num_display]
    img_matches = draw_matches(img_box, kp_box, img_scene, kp_scene, matches_top50,
                               f'orb_matches{save_suffix}.jpg')
    
    # RANSAC匹配图
    if len(inlier_matches) > 0:
        img_ransac = draw_matches(img_box, kp_box, img_scene, kp_scene, inlier_matches,
                                  f'ransac_matches{save_suffix}.jpg')
    else:
        img_ransac = None
    
    # 目标定位
    success = False
    img_result = None
    if H is not None and inlier_ratio > 0.2:
        img_result, _ = localize_object(img_scene, H, img_box.shape, f'target_localization{save_suffix}.jpg')
        success = True
    else:
        h, w = img_scene.shape[:2]
        img_result = np.ones((h, w, 3), dtype=np.uint8) * 255
        cv2.putText(img_result, "Localization Failed", (w//4, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(f'target_localization{save_suffix}.jpg', img_result)
    
    return {
        'nfeatures': nfeatures,
        'kp_box': len(kp_box),
        'kp_scene': len(kp_scene),
        'total_matches': total_matches,
        'inliers': inliers,
        'inlier_ratio': inlier_ratio,
        'success': success,
        'H': H,
        'img_box_kp': img_box_kp,
        'img_scene_kp': img_scene_kp,
        'img_matches': img_matches,
        'img_ransac': img_ransac,
        'img_result': img_result,
        'num_display': num_display
    }

# 主程序 
print("实验主题: 基于 OpenCV 的局部特征检测、描述与图像匹配")
print("="*80)

# 读取图像
img_box = cv2.imread('/home/ayn/project3/project6/box.jpg')
img_scene = cv2.imread('/home/ayn/project3/project6/box_in_scene.jpg')

if img_box is None or img_scene is None:
    print("错误: 无法读取图像文件，请确保 box.jpg 和 box_in_scene.jpg 存在")
    exit()

# 参数对比实验
nfeatures_list = [500, 1000, 2000]
exp_results = []
print("\n" + "="*80)
print("任务6: 参数对比实验 (改变 nfeatures)")
print("="*80)
print(f"{'nfeatures':<10} {'模板图关键点':<12} {'场景图关键点':<12} {'匹配数量':<10} {'RANSAC内点':<12} {'内点比例':<12} {'定位成功':<10}")
print("-"*80)

for nf in nfeatures_list:
    res = run_pipeline(img_box, img_scene, nf, save_suffix=f"_{nf}")
    exp_results.append(res)
    print(f"{res['nfeatures']:<10} {res['kp_box']:<12} {res['kp_scene']:<12} {res['total_matches']:<10} {res['inliers']:<12} {res['inlier_ratio']:<12.2%} {res['success']:<10}")

# 选择最佳参数（优先定位成功，再比较内点比例）
best_result = None
for res in exp_results:
    if res['success']:
        if best_result is None or res['inlier_ratio'] > best_result['inlier_ratio']:
            best_result = res
if best_result is None:
    best_result = max(exp_results, key=lambda x: x['inlier_ratio'])

best_nfeatures = best_result['nfeatures']
print(f"\n最佳参数选择: nfeatures = {best_nfeatures} (内点比例 {best_result['inlier_ratio']:.2%}, 定位{'成功' if best_result['success'] else '失败'})")

# 使用最佳参数重新运行（不添加后缀，覆盖最终输出文件）
print(f"\n使用最佳参数 nfeatures={best_nfeatures} 重新生成最终结果...")
final = run_pipeline(img_box, img_scene, best_nfeatures, save_suffix="")

print("\n最终结果（基于最佳参数）:")
print(f"box.jpg 关键点数量: {final['kp_box']}")
print(f"box_in_scene.jpg 关键点数量: {final['kp_scene']}")
print(f"描述子维度: {final['img_matches'].shape[-1] if final['img_matches'] is not None else 'N/A'}")
print(f"总匹配数量: {final['total_matches']}")
print(f"RANSAC内点数量: {final['inliers']}")
print(f"内点比例: {final['inlier_ratio']:.2%}")
print(f"Homography矩阵:\n{final['H']}")
if final['success']:
    print("目标定位: 成功 (绿色边框已绘制)")
else:
    print("目标定位: 失败")

# 输出所有保存的文件（强调1000和2000的结果）
print("\n" + "="*80)
print("已保存的所有结果文件（带后缀的为对应参数的结果）:")
print("  - box_keypoints_500.jpg, box_keypoints_1000.jpg, box_keypoints_2000.jpg")
print("  - scene_keypoints_500.jpg, scene_keypoints_1000.jpg, scene_keypoints_2000.jpg")
print("  - orb_matches_500.jpg, orb_matches_1000.jpg, orb_matches_2000.jpg")
print("  - ransac_matches_500.jpg, ransac_matches_1000.jpg, ransac_matches_2000.jpg")
print("  - target_localization_500.jpg, target_localization_1000.jpg, target_localization_2000.jpg")
print("最终输出（最佳参数，无后缀）:")
print("  - box_keypoints.jpg, scene_keypoints.jpg, orb_matches.jpg")
print("  - ransac_matches.jpg, target_localization.jpg, all_results.jpg")
print("="*80)

# 绘制最终汇总图（展示最佳参数的结果 + 参数对比表格）
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(final['img_box_kp'], cv2.COLOR_BGR2RGB))
plt.title(f'box.jpg 特征点 ({final["kp_box"]}个) - best nfeatures={best_nfeatures}')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(final['img_scene_kp'], cv2.COLOR_BGR2RGB))
plt.title(f'scene.jpg 特征点 ({final["kp_scene"]}个)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(final['img_matches'], cv2.COLOR_BGR2RGB))
plt.title(f'ORB匹配结果 (前{final["num_display"]}个, 总计{final["total_matches"]}个)')
plt.axis('off')

plt.subplot(2, 3, 4)
if final['img_ransac'] is not None:
    plt.imshow(cv2.cvtColor(final['img_ransac'], cv2.COLOR_BGR2RGB))
    plt.title(f'RANSAC内点匹配 ({final["inliers"]}/{final["total_matches"]}, {final["inlier_ratio"]:.1%})')
else:
    plt.imshow(np.ones((500, 500, 3), dtype=np.uint8) * 255)
    plt.title('RANSAC内点匹配 (无内点)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(final['img_result'], cv2.COLOR_BGR2RGB))
if final['success']:
    plt.title('目标定位结果 (绿色边框)')
else:
    plt.title('目标定位结果 (失败)')
plt.axis('off')

# 参数对比结果表格
plt.subplot(2, 3, 6)
plt.axis('tight')
plt.axis('off')
table_data = [[r['nfeatures'], r['kp_box'], r['kp_scene'], r['total_matches'], 
               r['inliers'], f"{r['inlier_ratio']:.1%}", '✓' if r['success'] else '✗'] 
              for r in exp_results]
table = plt.table(cellText=table_data, 
                  colLabels=['nfeatures', '模板关键点', '场景关键点', '匹配数', '内点数', '内点比例', '定位'],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.suptitle(f'ORB特征检测与图像匹配实验结果 (最佳参数: nfeatures={best_nfeatures})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('all_results.jpg', dpi=150, bbox_inches='tight')
plt.show()

print("\n程序执行完毕。")