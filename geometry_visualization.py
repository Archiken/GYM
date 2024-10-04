import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_on_frame(frame, results_1, results_2, frame_count):
    # 取得關鍵點位置
    keypoints_2 = results_2[0].keypoints.xy.cpu().numpy().tolist()[0]
    pos_p0 = np.array((int(keypoints_2[0][0]), int(keypoints_2[0][1])))
    pos_p1 = np.array((int(keypoints_2[1][0]), int(keypoints_2[1][1])))
    pos_p2 = np.array((int(keypoints_2[2][0]), int(keypoints_2[2][1])))
    pos_p3 = np.array((int(keypoints_2[3][0]), int(keypoints_2[3][1])))
    pos_m1 = np.array((int(keypoints_2[4][0]), int(keypoints_2[4][1])))
    pos_m2 = np.array((int(keypoints_2[5][0]), int(keypoints_2[5][1])))
    print("pos_p0:", pos_p0)
    print("pos_p1:", pos_p1)
    print("pos_p2:", pos_p2)
    print("pos_p3:", pos_p3)
    
    direction1 = pos_m2 - pos_m1
    direction_norm1 = direction1 / np.linalg.norm(direction1)

    direction2 = pos_p3 - pos_p2
    direction_norm2 = direction2 / np.linalg.norm(direction2)

    end_point = pos_p1 + direction_norm1 * 240
    end_point = tuple(end_point.astype(int))  # 轉換為整數，OpenCV 需要整數座標
    # 定義 3D 座標系統中的點
    P0 = np.array([0, 0, 0])
    P1 = np.array([0, 0, 275])
    P2 = np.array([144, 192, 275])
    P3 = np.array([144, 192, 0])
    print("P3:", P3)

    # 在影像上繪製點
    cv2.circle(frame, tuple(pos_p0), 5, (0, 0, 255), -1)  # P0 紅色
    cv2.circle(frame, tuple(pos_p1), 5, (0, 255, 0), -1)  # P1 綠色
    cv2.circle(frame, tuple(pos_p2), 5, (0, 255, 0), -1)  # P2 藍色
    cv2.circle(frame, tuple(pos_p3), 5, (255, 0, 0), -1)  # P3 黃色

    # 在影像上繪製連線
    cv2.line(frame, tuple(pos_p0), tuple(pos_p1), (0, 0, 255), 2)
    cv2.line(frame, tuple(pos_p1), tuple(pos_p2), (0, 255, 0), 2)
    cv2.line(frame, tuple(pos_p2), tuple(pos_p3), (255, 0, 0), 2)


    keypoints_1 = results_1[0].keypoints.xy.cpu().numpy().tolist()[0]
    R_hand = np.array((int(keypoints_1[8][0]), int(keypoints_1[8][1])))

    points = {
    'head': np.array((int(keypoints_1[0][0]), int(keypoints_1[0][1]))),
    'CT': np.array((int(keypoints_1[1][0]), int(keypoints_1[1][1]))),
    'TL': np.array((int(keypoints_1[2][0]), int(keypoints_1[2][1]))),
    'R_hip': np.array((int(keypoints_1[3][0]), int(keypoints_1[3][1]))),
    'R_knee': np.array((int(keypoints_1[4][0]), int(keypoints_1[4][1]))),
    'R_ankle': np.array((int(keypoints_1[5][0]), int(keypoints_1[5][1]))),
    'R_shoulder': np.array((int(keypoints_1[6][0]), int(keypoints_1[6][1]))),
    'R_elbow': np.array((int(keypoints_1[7][0]), int(keypoints_1[7][1]))),
    'R_hand': np.array((int(keypoints_1[8][0]), int(keypoints_1[8][1])))}
    
    # 找到手的投影點 hand_proj
    # 計算直線方程的斜率和截距
    if (pos_p2[0] - pos_p1[0]) != 0:
        m = (pos_p2[1] - pos_p1[1]) / (pos_p2[0] - pos_p1[0])
        b = pos_p1[1] - m * pos_p1[0]

        # 找到 R_hand x 值在直線上的 y 坐標
        hand_proj_y = m * R_hand[0] + b
        hand_proj = np.array([R_hand[0], int(hand_proj_y)])
    else:
        # 當直線垂直時，y 坐標可以任意，x 坐標為 pos_p1[0]（或 pos_p2[0]）
        hand_proj = np.array([pos_p1[0], R_hand[1]])
        
    print("R_hand:", R_hand)
    print("hand_proj:", hand_proj)
    Dis = R_hand[1] - hand_proj[1]
    print("Distance:", Dis)
    # 定義顏色
    colors_2d = {
        'head': (65, 66, 136),
        'CT': (65, 66, 136),
        'TL': (65, 66, 136),
        'R_hip': (95, 180, 156),
        'R_knee': (95, 180, 156),
        'R_ankle': (95, 180, 156),
        'R_shoulder': (222, 239, 183),
        'R_elbow': (222, 239, 183),
        'R_hand': (222, 239, 183),
    }
    
    # 在畫面上繪製每個點
    for name, point in points.items():
        color = colors_2d.get(name, (0, 0, 0))  # 從顏色字典中取得顏色
        cv2.circle(frame, tuple(point), 5, color, -1)  # 使用指定顏色繪製點

    cv2.circle(frame, hand_proj, 5, (0, 0, 0), -1)  # 5 為半徑，黑色的圓圈

    # 繪製指定的點之間的連接線
    connections = [
        ('head', 'CT'),
        ('CT', 'TL'),
        ('CT', 'R_shoulder'),
        ('R_shoulder', 'R_elbow'),
        ('R_elbow', 'R_hand'),
        ('TL', 'R_hip'),
        ('R_hip', 'R_knee'),
        ('R_knee', 'R_ankle'),
    ]

    # 繪製連接線
    for start, end in connections:
        start_point = tuple(points[start])
        end_point = tuple(points[end])
        cv2.line(frame, start_point, end_point, (0, 0, 0), 2)  # 使用黑色線條，線寬2px

    # 判斷 Dis 是否小於 -20
    if Dis < -20:
        # 當 Dis < -20 時，將 rot_origin 設為 P1 和 P2 的中點
        rot_origin = (P1 + P2) / 2
    else:
        # 計算 rot_origin，根據手的投影點在 P1 和 P2 之間的位置
        t = (hand_proj[0] - pos_p1[0]) / (pos_p2[0] - pos_p1[0])
        rot_origin = P1 + t * (P2 - P1)
    print("rot_origin:", rot_origin)
        

    # ---- 在畫面右下角繪製 3D 點與連線 ---- #
    # 設置右下角繪圖區域 (600x600)
    sub_frame_height = 600
    sub_frame_width = 600
    top_right_x = frame.shape[1] - sub_frame_width
    top_right_y = 0
    top_left_x = 0
    top_left_y = 0

    # 計算平面的法向量 (P2 - P1)
    normal_vector = P2 - P1

    # 找到平面內的兩個方向向量（與法向量垂直）
    # 我們可以取一個隨機向量與法向量做叉積來確保垂直
    v1 = np.cross(normal_vector, [1, 0, 0])  # 叉積計算第一個方向向量
    v1 = v1 / np.linalg.norm(v1)  # 歸一化，使向量單位長度
    v2 = np.cross(normal_vector, v1)  # 第二個垂直向量
    v2 = v2 / np.linalg.norm(v2)  # 對 v2 進行歸一化

    # 計算正方形的四個頂點
    side_length = 500
    half_side = side_length / 2
    corner1 = rot_origin + half_side * v1 + half_side * v2
    corner2 = rot_origin + half_side * v1 - half_side * v2
    corner3 = rot_origin - half_side * v1 - half_side * v2
    corner4 = rot_origin - half_side * v1 + half_side * v2

    # 定義正方形的頂點
    rot_square = np.array([corner1, corner2, corner3, corner4])
    print("rot_square:", rot_square)
    # 定義每組關節的顏色
    colors = {
        'head': (65/255, 66/255, 136/255),
        'CT': (65/255, 66/255, 136/255),
        'TL': (65/255, 66/255, 136/255),
        'R_hip': (95/255, 180/255, 156/255),
        'R_knee': (95/255, 180/255, 156/255),
        'R_ankle': (95/255, 180/255, 156/255),
        'R_shoulder': (222/255, 239/255, 183/255),
        'R_elbow': (222/255, 239/255, 183/255),
        'R_hand': (222/255, 239/255, 183/255),
    }
    # 計算每個點需要移動的 d1 和 d2
    # 計算每個點的投影
    projections = {}
    for name, point in points.items():
        # 計算相對位置向量
        relative_vec = point - hand_proj
        
        # 計算在 direction_norm1 上的投影
        d1 = np.dot(relative_vec, direction_norm1)
        
        # 計算在 direction_norm2 上的投影
        d2 = np.dot(relative_vec, direction_norm2)
        
        # 計算每個點在 rot_square 平面上的投影
        projection_point = rot_origin - d1 * v2/2 + d2 * v1/2
        # projection_point = rot_origin - d1 * v2 + d2 * v1
        projections[name] = projection_point
        print(f"{name}_projection:", projection_point)
    # 創建 Matplotlib 圖像
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d') 

    # 創建 Matplotlib 圖像（左上角）
    fig_left = plt.figure(figsize=(6, 6), dpi=100)
    ax_left = fig_left.add_subplot(111, projection='3d')

    # 繪製正方形（rot_square）
    square = Poly3DCollection([rot_square], facecolors=(219/255, 254/255, 184/255), 
                              edgecolors='k', linewidths=1, alpha=0.4)
    ax.add_collection3d(square)

    # 設置淺綠色平面（rot_square）
    square_left = Poly3DCollection([rot_square], facecolors=(219/255, 254/255, 184/255), 
                                edgecolors='k', linewidths=1, alpha=0.2)
    ax_left.add_collection3d(square_left)

    # 設定軸的範圍
    ax.set_xlim([-135, 297])
    ax.set_ylim([-129, 321])
    ax.set_zlim([0, 550])
    # 計算方位角，隨幀數旋轉
    rotation_speed = 0.6  # 控制旋轉速度，值越大旋轉越快
    azim_angle = frame_count * rotation_speed
    # 設置 3D 圖形的視角
    ax.view_init(elev=10, azim=azim_angle)  # 設定仰角和方位角
    ax_left.view_init(elev=0, azim=45)

    # 繪製 3D 點    
    ax.scatter(*P0, c='r', label='P0')
    ax.scatter(*P1, c='g', label='P1')
    ax.scatter(*P2, c='g', label='P2')
    ax.scatter(*P3, c='b', label='P3')
    ax.scatter(*rot_origin, c='k', label='rot_origin', marker='x')  # 使用黑色的 'x' 標記這個點
 
    # 繪製 P0-P1, P1-P2, P2-P3 的連線
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], c='r')
    ax_left.plot([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], c='r')
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], c='g')
    ax_left.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], c='g')
    ax.plot([P2[0], P3[0]], [P2[1], P3[1]], [P2[2], P3[2]], c='b')
    ax_left.plot([P2[0], P3[0]], [P2[1], P3[1]], [P2[2], P3[2]], c='b')

    # 繪製九個點在正方形平面上的投影，並根據名稱設置顏色
    for name, proj_point in projections.items():
        color = colors.get(name, (0, 0, 0))  # 默認黑色
        ax.scatter(*proj_point, color=color, label=f'{name}_projection')
        ax_left.scatter(*proj_point, color=color, label=f'{name}_projection')

    for start, end in connections:
        start_point = projections[start]
        end_point = projections[end]
        # 用黑色線條連接兩個點
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], color='k')
        ax_left.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], color='k')

    # 設置標籤
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 將 Matplotlib 圖像保存到內存中
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close()

    # 移除 x, y, z 軸標籤和刻度
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_zticks([])
    ax_left.set_xlabel('')
    ax_left.set_ylabel('')
    ax_left.set_zlabel('')

    # 將 Matplotlib 圖像保存到內存中（左上角）
    buf_left = io.BytesIO()
    plt.savefig(buf_left, format='png', bbox_inches='tight', pad_inches=0.1)
    buf_left.seek(0)
    plt.close()

    # 使用 PIL 將內存中的 PNG 圖像轉換為 OpenCV 格式
    image_pil = Image.open(buf)
    plot_img = np.array(image_pil)
    image_pil_left = Image.open(buf_left)
    plot_img_left = np.array(image_pil_left)

    # 將圖像從 RGB 轉換為 BGR，因為 OpenCV 使用 BGR 色彩空間
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
    plot_img_left = cv2.cvtColor(plot_img_left, cv2.COLOR_RGB2BGR)

    # 調整 Matplotlib 圖像的大小
    plot_img_resized = cv2.resize(plot_img, (sub_frame_width, sub_frame_height))
    plot_img_resized_left = cv2.resize(plot_img_left, (sub_frame_width, sub_frame_height))
    
    # 插入到畫面的右上角
    frame[top_right_y:top_right_y + sub_frame_height, top_right_x:top_right_x + sub_frame_width] = plot_img_resized
    frame[top_left_y:top_left_y + sub_frame_height, top_left_x:top_left_x + sub_frame_width] = plot_img_resized_left

    return frame
