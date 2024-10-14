import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_keypoint(frame, pos, label, color=(255, 255, 0), font_size=1.2, font_thickness=2, offset=(10, 30), draw_circle=True):
    """ 在圖像上繪製關鍵點並標註坐標 """
    if draw_circle:
        cv2.circle(frame, pos, 5, color, -1)  # 繪製圓點
    cv2.putText(frame, label, (pos[0] + offset[0], pos[1] + offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness + 3, cv2.LINE_AA) #畫黑色外框
    cv2.putText(frame, label, (pos[0] + offset[0], pos[1] + offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness, cv2.LINE_AA)
    

hip_buffer = [] 
ankle_buffer = []
buffer_size = 5

def draw_on_frame(frame, results_1, results_2, frame_count, rot_origin_last, rot_proj_last, R_hip_last_angle, R_ankle_last_angle):
    # 取得關鍵點位置
    keypoints_2 = results_2[0].keypoints.xy.cpu().numpy().tolist()[0]
    pos_p0 = np.array((int(keypoints_2[0][0]), int(keypoints_2[0][1])))
    pos_p1 = np.array((int(keypoints_2[1][0]), int(keypoints_2[1][1])))
    pos_p2 = np.array((int(keypoints_2[2][0]), int(keypoints_2[2][1])))
    pos_p3 = np.array((int(keypoints_2[3][0]), int(keypoints_2[3][1])))
    pos_m1 = np.array((int(keypoints_2[4][0]), int(keypoints_2[4][1])))
    pos_m2 = np.array((int(keypoints_2[5][0]), int(keypoints_2[5][1])))
    # print(f"Distance between pos_p0 and pos_p1: {np.linalg.norm(pos_p1 - pos_p0)}")
    # print("pos_p0:", pos_p0)
    # print("pos_p1:", pos_p1)
    # print("pos_p2:", pos_p2)
    # print("pos_p3:", pos_p3)
    
    direction1 = pos_m2 - pos_m1
    direction_norm1 = direction1 / np.linalg.norm(direction1)

    direction2 = pos_p3 - pos_p2
    direction_norm2 = direction2 / np.linalg.norm(direction2)

    # 定義 3D 座標系統中的點
    P0 = np.array([0, 0, 0])
    P1 = np.array([0, 0, 275])
    P2 = np.array([144, 192, 275])
    P3 = np.array([144, 192, 0])
    # print("P3:", P3)

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
    m = (pos_p2[1] - pos_p1[1]) / (pos_p2[0] - pos_p1[0])
    b = pos_p1[1] - m * pos_p1[0]

    # 判斷 R_hand[0] 是否在 pos_p1[0] 和 pos_p2[0] 之間
    if R_hand[0] < pos_p1[0] or R_hand[0] > pos_p2[0]:
        # 如果不在範圍內，將 hand_proj 設置為 pos_p1 和 pos_p2 的中點
        hand_proj = (pos_p1 + pos_p2) / 2
    else: 
        # 找到 R_hand x 值在直線上的 y 坐標
        hand_proj_y = m * R_hand[0] + b
        hand_proj = np.array([R_hand[0], int(hand_proj_y)])


    hand_proj = hand_proj.astype(int)
    # print("R_hand:", R_hand)
    # print("hand_proj:", hand_proj)
    Dis = R_hand[1] - hand_proj[1]
    # 标注 rot_origin 的 3D 坐标
    # draw_keypoint(frame, (5, 900), f"Dis: {Dis}", color=(255, 255, 255), draw_circle=False)
    # print("Distance:", Dis)
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

    # Dis若<-25或>25 以單槓中點為旋轉原點
    if Dis < -25 or Dis > 25:
        rot_origin = (P1 + P2) / 2
        rot_origin_proj = (pos_p1 + pos_p2)/2

    else:
        # 計算 rot_origin，根據手的投影點在 P1 和 P2 之間的位置
        t = (hand_proj[0] - pos_p1[0]) / (pos_p2[0] - pos_p1[0])
        rot_origin = P1 + t * (P2 - P1)
        rot_origin_proj = pos_p1 + t * (pos_p2 - pos_p1)

    if rot_origin_last is not None:
        # 檢查 rot_origin_last[0] 是否不為 72 且與當前幀的 rot_origin_x 差異超過 25
        if rot_origin[0] - rot_origin_last[0] > 20 or rot_origin[0] - rot_origin_last[0] < -20:
            rot_origin = (rot_origin_last + rot_origin) / 2
            rot_origin_proj = (rot_proj_last + rot_origin_proj) / 2


    
    rot_origin_proj = rot_origin_proj.astype(int)  # 將數值轉換為整數
    # draw_keypoint(frame, (5, 945), f"rot_origin: {rot_origin}", color=(0, 255, 255), draw_circle=False)
    # if rot_origin_last is not None:
    #     draw_keypoint(frame, (5, 990), f"origin-last: {rot_origin[0] - rot_origin_last[0]}", color=(255, 0, 255), draw_circle=False)
    # else:
    #     draw_keypoint(frame, (5, 990), "origin-last: N/A", color=(255, 0, 255), draw_circle=False)
    cv2.circle(frame, tuple(rot_origin_proj), 5, (255, 0, 0), -1)  
    # print("rot_origin:", rot_origin)
        

    # ---- 在畫面右下角繪製 3D 點與連線 ---- #
    # 設置右下角繪圖區域 (600x600)
    sub_frame_height = 600
    sub_frame_width = 600

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
    # print("rot_square:", rot_square)
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
        relative_vec = point - rot_origin_proj
        
        # 計算在 direction_norm1 上的投影
        d1 = np.dot(relative_vec, direction_norm1)
        
        # 計算在 direction_norm2 上的投影
        d2 = np.dot(relative_vec, direction_norm2)
        
        # 計算每個點在 rot_square 平面上的投影
        projection_point = rot_origin + d1 * v2/2 + d2 * v1/2

        projections[name] = projection_point
        # print(f"{name}_projection:", projection_point)
    # 創建 Matplotlib 圖像
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d') 

    # 繪製正方形（rot_square）
    square = Poly3DCollection([rot_square], facecolors=(219/255, 254/255, 184/255), 
                              edgecolors='k', linewidths=1, alpha=0.4)
    ax.add_collection3d(square)

    # 設定軸的範圍
    ax.set_xlim([-172, 378])
    ax.set_ylim([-158, 392])
    ax.set_zlim([0, 550])
    # 計算方位角，隨幀數旋轉
    rotation_speed = 0.6  # 控制旋轉速度，值越大旋轉越快
    azim_angle = frame_count * rotation_speed
    # 設置 3D 圖形的視角
    ax.view_init(elev=10, azim=azim_angle)  # 設定仰角和方位角

    # 繪製 3D 點    
    ax.scatter(*P0, c='r', label='P0')
    ax.scatter(*P1, c='g', label='P1')
    ax.scatter(*P2, c='g', label='P2')
    ax.scatter(*P3, c='b', label='P3')
    ax.scatter(*rot_origin, c='k', label='rot_origin', marker='x')  # 使用黑色的 'x' 標記這個點
 
    # 繪製 P0-P1, P1-P2, P2-P3 的連線
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]], [P0[2], P1[2]], c='r')
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], c='g')
    ax.plot([P2[0], P3[0]], [P2[1], P3[1]], [P2[2], P3[2]], c='b')

    # 繪製九個點在正方形平面上的投影，並根據名稱設置顏色
    for name, proj_point in projections.items():
        color = colors.get(name, (0, 0, 0))  # 默認黑色
        ax.scatter(*proj_point, color=color, label=f'{name}_projection')

    for start, end in connections:
        start_point = projections[start]
        end_point = projections[end]
        # 用黑色線條連接兩個點
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], color='k')

    # 處理右上角圖像
    # 添加需要的繪圖內容到ax
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    plot_img = np.array(Image.open(buf))
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
    plot_img_resized = cv2.resize(plot_img, (sub_frame_width, sub_frame_height))
    frame[0:sub_frame_height, frame.shape[1]-sub_frame_width:frame.shape[1]] = plot_img_resized

    
    #-------左上角角度-------
    # 左上角原點 (300, 300)
    origin = np.array([300, 250])
    # 設定幀率
    fps = 30  # 幀率 (frames per second)
    time_per_frame = 1 / fps  # 每幀的時間

    # 創建 overlay 並繪製淺綠色矩形
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (600, 500), (184, 254, 219), -1)

    # 混合 overlay 和 frame，實現透明效果
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    # 在淺綠色矩形外圍加一個黑色邊框
    cv2.rectangle(frame, (0, 0), (600, 500), (0, 0, 0), 2)  # 2 為邊框寬度
    # 從 (300, 300) 到 (300, 600) 繪製紅線
    cv2.line(frame, (300, 250), (300, 500), (0, 0, 255), 2)  # 紅線，線寬 2

    # 計算每個投影點在 2D 畫面上的位置
    projected_points_2D = {}
    for name, proj_point in points.items():
        # 以 (300, 300) 為原點，根據 d1 和 d2 確定每個點的位置
        relative_vec = proj_point - rot_origin_proj
        d1 = np.dot(relative_vec, direction_norm1)
        d2 = np.dot(relative_vec, direction_norm2)
        x = int(origin[0] - d1/2.1)  # 水平方向的位移
        y = int(origin[1] + d2/2.1)  # 垂直方向的位移
        projected_points_2D[name] = (x, y)
    
    R_hip = projected_points_2D['R_hip']
    R_ankle = projected_points_2D['R_ankle']

    # 計算當前幀 R_hip 相對於 origin 的角度
    hip_vector = np.array(R_hip) - origin
    R_hip_angle = np.arctan2(hip_vector[1], hip_vector[0])  # 反正切計算角度，弧度制
    R_hip_angle = np.degrees(R_hip_angle)  # 將弧度轉換為度數
    # 將角度範圍從 [-180, 180] 轉換到 [0, 360]
    if R_hip_angle < 0:
        R_hip_angle += 360

    hip_vector = np.array(R_ankle) - origin
    R_ankle_angle = np.arctan2(hip_vector[1], hip_vector[0])  # 反正切計算角度，弧度制
    R_ankle_angle = np.degrees(R_ankle_angle)  # 將弧度轉換為度數
    if R_ankle_angle < 0:
        R_ankle_angle += 360

    hip_ang_velocity = 0
    hip_smoothed = 0
    ankle_ang_velocity = 0
    ankle_smoothed = 0

    # 如果這不是第一幀，則計算角速度
    if R_hip_last_angle is not None:
        # 計算髖關節角度變化
        hip_change_deg = R_hip_angle - R_hip_last_angle

        # 防止角度跳變，可以將角度歸一化到 [-180, 180] 範圍
        hip_change_deg = (hip_change_deg + 180) % 360 - 180

        # 計算角速度 (角度變化除以每幀的時間)
        hip_ang_velocity = hip_change_deg / time_per_frame
        hip_ang_velocity = round(hip_ang_velocity, 1)

        # 將角速度添加到緩衝區
        hip_buffer.append(hip_ang_velocity)
        if len(hip_buffer) > buffer_size:
            hip_buffer.pop(0)  # 移除最舊的角速度數據
        # 平滑角速度，取緩衝區內所有角速度的平均值
        hip_smoothed = round(np.mean(hip_buffer), 1)

        # draw_keypoint(frame, (5, 545), f"Angular velocity: {hip_ang_velocity} degrees/s", color=(255, 255, 255), draw_circle=False)
        # draw_keypoint(frame, (5, 590), f"smoothed_velocity3: {hip_smoothed} degrees/s", color=(255, 255, 255), draw_circle=False)
        # print(f"Angular velocity of R_hip: {hip_ang_velocity} degrees/s")
        # 計算踝關節角度變化
        ankle_change_deg = R_ankle_angle - R_ankle_last_angle

        # 防止角度跳變，可以將角度歸一化到 [-180, 180] 範圍
        ankle_change_deg = (ankle_change_deg + 180) % 360 - 180

        # 計算角速度 (角度變化除以每幀的時間)
        ankle_ang_velocity = ankle_change_deg / time_per_frame
        ankle_ang_velocity = round(ankle_ang_velocity, 1)

        # 將角速度添加到緩衝區
        ankle_buffer.append(ankle_ang_velocity)
        if len(ankle_buffer) > buffer_size:
            ankle_buffer.pop(0)  # 移除最舊的角速度數據
        # 平滑角速度，取緩衝區內所有角速度的平均值
        ankle_smoothed = round(np.mean(ankle_buffer), 1)

    # 更新上一幀的角度為當前幀的角度
    R_hip_last_angle = R_hip_angle
    R_ankle_last_angle = R_ankle_angle
    
    # 定义每组关节的颜色（BGR 格式）
    colors_bgr = {
        'head': (136, 66, 65),
        'CT': (136, 66, 65),
        'TL': (136, 66, 65),
        'R_hip': (156, 180, 95),
        'R_knee': (156, 180, 95),
        'R_ankle': (156, 180, 95),
        'R_shoulder': (183, 239, 222),
        'R_elbow': (183, 239, 222),
        'R_hand': (183, 239, 222),}
    
    # 在图像上绘制每个点
    for name, (x, y) in projected_points_2D.items():
        color = colors_bgr.get(name, (0, 0, 0))  # 默认黑色
        cv2.circle(frame, (x, y), 5, color, -1)

    # 绘制连接线
    for start, end in connections:
        start_point = projected_points_2D[start]
        end_point = projected_points_2D[end]
        cv2.line(frame, start_point, end_point, (0, 0, 0), 2)  # 黑色线条，线宽 2

    return frame, rot_origin,  rot_origin_proj, R_hip_last_angle, hip_smoothed, R_ankle_last_angle, ankle_smoothed

hip_smoothed_history = []
ankle_smoothed_history = []
display_buffer = 70
def draw_on_frame_with_plot(frame, hip_smoothed, ankle_smoothed):
    # 將 hip_smoothed 添加到歷史數據緩衝區
    hip_smoothed_history.append(abs(hip_smoothed))  # 保存絕對值
    if len(hip_smoothed_history) > display_buffer:
        hip_smoothed_history.pop(0)  # 移除最舊的數據
    # 將 ankle_smoothed 添加到歷史數據緩衝區
    ankle_smoothed_history.append(abs(ankle_smoothed))  # 保存絕對值
    if len(ankle_smoothed_history) > display_buffer:
        ankle_smoothed_history.pop(0)  # 移除最舊的數據

    # 創建 Matplotlib 圖像
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot(hip_smoothed_history, color='b', label='Hip')  # 繪製折線圖
    ax.plot(ankle_smoothed_history, color='r', label='Ankle')  # 繪製折線圖

    # 設定軸範圍
    ax.set_xlim(0, display_buffer)
    ax.set_ylim(0, 1000)  # 縱軸最大值 1000
    # ax.set_yticks(np.arange(100, 1001, 100))
    ax.set_yticks([250, 500, 750, 1000])
    ax.tick_params(axis='y', labelsize=15)

    ax.set_title("Rotation Speed (deg/s)", fontsize=15)

    # 移除 x  軸的標籤
    ax.set_xticks([])
    # 添加圖例（標示顏色）
    ax.legend(loc="upper left", fontsize=15)  # 將圖例放在左上角

    # 將 Matplotlib 圖像保存到內存中
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)

    # 使用 PIL 讀取圖像，並轉換為 OpenCV 格式
    plot_img = np.array(Image.open(buf))
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    # 調整圖像大小為 400x300
    plot_img_resized = cv2.resize(plot_img, (400, 300))

    # 將折線圖嵌入到原始影像的左下角
    frame[frame.shape[0]-300:frame.shape[0], 0:400] = plot_img_resized

    return frame
