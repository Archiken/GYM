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
    

COG_buffer = [] 
buffer_size = 3
COG_flight_height = []
COG_smoothed_history = []
COG_angle_history = []
display_buffer = 70

def draw_on_frame(frame, results_1, results_2, frame_count, rot_origin_last, rot_proj_last,
                  COG_last_angle, flight_last_vel, COG_last_height, COG_last_cal_height,
                  frame_data_table, COG_flight_height_frames):
    # 取得關鍵點位置
    keypoints_2 = results_2[0].keypoints.xy.cpu().numpy().tolist()[0]
    pos_p0 = np.array((int(keypoints_2[0][0]), int(keypoints_2[0][1])))
    pos_p1 = np.array((int(keypoints_2[1][0]), int(keypoints_2[1][1])))
    pos_p2 = np.array((int(keypoints_2[2][0]), int(keypoints_2[2][1])))
    pos_p3 = np.array((int(keypoints_2[3][0]), int(keypoints_2[3][1])))
    pos_m1 = np.array((int(keypoints_2[4][0]), int(keypoints_2[4][1])))
    pos_m2 = np.array((int(keypoints_2[5][0]), int(keypoints_2[5][1])))
    
    direction1 = pos_m2 - pos_m1
    direction_norm1 = direction1 / np.linalg.norm(direction1)

    direction2 = pos_p3 - pos_p2
    direction_norm2 = direction2 / np.linalg.norm(direction2)

    # 定義 3D 座標系統中的點
    P0 = np.array([0, 0, 0])
    P1 = np.array([0, 0, 275])
    P2 = np.array([144, 192, 275])
    P3 = np.array([144, 192, 0])

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

    # 定義質量百分比
    mass_percentage = {
        'head': 0.082,      # 頭部
        'torso': 0.4684,    # 軀幹
        'uarm': 0.065,      # 上臂
        'larm': 0.036,      # 前臂
        'hand': 0.013,      # 手
        'thigh': 0.21,      # 大腿
        'calf': 0.095,      # 小腿
        'foot': 0.0286      # 腳
    }

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
    
    # 計算各個肢段的重心位置
    head = points['head']
    torso = points['TL']
    uarm = (points['R_shoulder'] + points['R_elbow']) / 2
    larm = (points['R_elbow'] + points['R_hand']) / 2
    hand = points['R_hand']
    thigh = (points['R_hip'] + points['R_knee']) / 2
    calf = (points['R_knee'] + points['R_ankle']) / 2
    foot = points['R_ankle']

    # 加權平均計算 COG
    COG_x = (mass_percentage['head'] * head[0] +
            mass_percentage['torso'] * torso[0] +
            mass_percentage['uarm'] * uarm[0] +
            mass_percentage['larm'] * larm[0] +
            mass_percentage['hand'] * hand[0] +
            mass_percentage['thigh'] * thigh[0] +
            mass_percentage['calf'] * calf[0] +
            mass_percentage['foot'] * foot[0]) / sum(mass_percentage.values())

    COG_y = (mass_percentage['head'] * head[1] +
            mass_percentage['torso'] * torso[1] +
            mass_percentage['uarm'] * uarm[1] +
            mass_percentage['larm'] * larm[1] +
            mass_percentage['hand'] * hand[1] +
            mass_percentage['thigh'] * thigh[1] +
            mass_percentage['calf'] * calf[1] +
            mass_percentage['foot'] * foot[1]) / sum(mass_percentage.values())

    COG = np.array([int(COG_x), int(COG_y)])  # 最終 COG 取整數
    points['COG'] = COG


    
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
    Dis = R_hand[1] - hand_proj[1]
    # 标注 rot_origin 的 3D 坐标
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
        'COG': (0,0,220)
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
    cv2.circle(frame, tuple(rot_origin_proj), 5, (255, 0, 0), -1)  
        

    # ---- 在畫面右上角繪製 3D 點與連線 ---- #
    # 設置右上角繪圖區域
    sub_frame_height = 500
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
        'COG': (220/255, 0/255, 0/255),
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
    fig = plt.figure(figsize=(6, 5), dpi=100)
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
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    plot_img = np.array(Image.open(buf))
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
    plot_img_resized = cv2.resize(plot_img, (sub_frame_width, sub_frame_height))
    frame[0:sub_frame_height, frame.shape[1]-sub_frame_width:frame.shape[1]] = plot_img_resized

    
    #-------左上角角度-------
    # 左上角原點 (300, 250)
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
    cv2.rectangle(frame, (0, 0), (600, 500), (0, 0, 0), 3)  # 3 為邊框寬度
    # 從 (300, 250) 到 (300, 500) 繪製紅線
    cv2.line(frame, (300, 250), (300, 500), (0, 0, 255), 2)  # 紅線，線寬 2

    # 計算每個投影點在 2D 畫面上的位置
    projected_points_2D = {}
    for name, proj_point in points.items():
        # 以 (300, 250) 為原點，根據 d1 和 d2 確定每個點的位置
        relative_vec = proj_point - rot_origin_proj
        d1 = np.dot(relative_vec, direction_norm1)
        d2 = np.dot(relative_vec, direction_norm2)
        x = int(origin[0] - d1/2.2)  # 水平方向的位移
        y = int(origin[1] + d2/2.2)  # 垂直方向的位移
        projected_points_2D[name] = (x, y)
    
    COG = projected_points_2D['COG']
    Dis_COG = np.linalg.norm(np.array(COG) - origin)

    COG_vector = np.array(COG) - origin
    COG_distance = np.linalg.norm(COG_vector)
    
    COG_angle = np.arctan2(COG_vector[1], COG_vector[0])
    COG_angle = np.degrees(COG_angle)
    if COG_angle < 0:
        COG_angle += 360

    # draw_keypoint(frame, (1200, 900), f"COG_angle: {COG_angle}", color=(0, 255, 255), draw_circle=False)
    # 儀表板設定
    center = (540, 950)
    radius = 120
    pointer_length = 115
    color_yellow = (99, 213, 255)
    
    # 繪製儀表板背景
    cv2.rectangle(frame, (400, 780), (680, 1080), (255, 255, 255), -1)
    cv2.circle(frame, center, radius, (0, 0, 0), 2)
    cv2.putText(frame, 'COG angular position', (418 , 805),
                cv2.FONT_HERSHEY_SIMPLEX, 0.73, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, '3D Estimation', (1515 , 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, '2D Projection', (170 , 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    draw_keypoint(frame, (680 , 5), "2024 Baku World Cup", color=(255, 255, 255), draw_circle=False)
    draw_keypoint(frame, (715 , 40), "TANG_Chia-Hung", color=(255, 255, 255), draw_circle=False)
    
    # 填充黃色區域
    overlay = frame.copy()

    # 計算 300 度到 360 度的圓餅形區域
    cv2.ellipse(overlay, center, (radius, radius), 0, -60, 0, color_yellow, -1)
    # 計算 180 度到 240 度的圓餅形區域
    cv2.ellipse(overlay, center, (radius, radius), 0, -180, -120, color_yellow, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 繪製輔助線 (240度和300度)
    for angle in [240, 300]:
        end_x = int(center[0] + radius * np.cos(np.radians(360 - angle)))
        end_y = int(center[1] - radius * np.sin(np.radians(360 - angle)))
        cv2.line(frame, center, (end_x, end_y), (0, 0, 0), 1)
    
    # 標記角度
    for angle in range(0, 360, 30):
        x = int(center[0] + (radius + 5) * np.cos(np.radians(360 - angle)))
        y = int(center[1] - (radius + 5) * np.sin(np.radians(360 - angle)))
        cv2.circle(frame, (x, y), 2, (0, 0, 0), -1)
    
    # 繪製指針
    angle_rad = np.radians(360 - COG_angle)
    end_x = int(center[0] + pointer_length * np.cos(angle_rad))
    end_y = int(center[1] - pointer_length * np.sin(angle_rad))
    cv2.line(frame, center, (end_x, end_y), (0, 0, 220), 2)
    
    COG_ang_velocity = 0
    COG_smoothed = 0
    COG_tangential_vel = 0
    flight_vel = 0
    COG_cal_height = 0
    COG_height = 0
    COG_vertical_vel = 0

    if COG_last_angle is not None:
        COG_change_deg = COG_angle - COG_last_angle
        COG_change_deg = (COG_change_deg +180) % 360 - 180

        COG_ang_velocity = COG_change_deg / time_per_frame
        COG_ang_velocity = round(COG_ang_velocity, 1)

        COG_buffer.append(COG_ang_velocity)
        if len(COG_buffer) > buffer_size:
            COG_buffer.pop(0)
        COG_smoothed = round(abs(np.mean(COG_buffer)), 1)

        # 計算 COG 的瞬時速度
        COG_tangential_vel = COG_smoothed * (np.pi / 180) * COG_distance * 1.08  # 單位取決於角速度的單位（如度/秒或弧度/秒）
        COG_vertical_vel = round(abs(COG_tangential_vel * np.cos(np.radians(COG_angle))), 1)
        last_update_frame = None

        if flight_last_vel is not None:
            # 條件一：COG_last_angle 小於 40 度，且 COG_angle 大於 330 度
            if COG_last_angle < 40 and COG_angle > 330:
                flight_vel = COG_vertical_vel

            # 條件二：COG_last_angle 小於180 度，且 COG_angle 大於 180 度
            elif COG_last_angle < 180 and COG_angle > 180:
                flight_vel = COG_vertical_vel

            # 新條件三：COG_angle 大於 300，且 COG_angle 小於 COG_last_angle(逆轉)，且 flight_last_vel 小於 COG_vertical_vel
            elif COG_angle > 300 and COG_angle < COG_last_angle and flight_last_vel < COG_vertical_vel:
                flight_vel = COG_vertical_vel

            # 新條件四：COG_angle 小於 240，且 COG_angle 大於 COG_last_angle(順轉)，且 flight_last_vel 小於 COG_vertical_vel
            elif 180 <= COG_angle < 240 and COG_angle > COG_last_angle and flight_last_vel < COG_vertical_vel:
                flight_vel = COG_vertical_vel

            else:
                flight_vel = flight_last_vel
            
            # 條件：計算 COG_cal_height 當 COG_angle < 300 且 COG_angle < COG_last_angle
            if COG_angle < 300 and 270 < COG_angle < COG_last_angle:
                COG_cal_height = (flight_vel ** 2) / (19.6 * 1.08)
                last_update_frame = frame_count  # 記錄此時的 frame_count

            # 條件：計算 COG_cal_height 當 COG_angle > 240 且 COG_angle > COG_last_angle
            elif (COG_angle > 240 and COG_angle > COG_last_angle >= 180) or (Dis_COG > 150):
                COG_cal_height = (flight_vel ** 2) / (19.6 * 1.08)
                last_update_frame = frame_count  # 記錄此時的 frame_count

            else:
                COG_cal_height = COG_last_cal_height

            # draw_keypoint(frame, (1200, 500), f"COG_tangential_vel: {COG_tangential_vel}", color=(255, 255, 255), draw_circle=False)
            # draw_keypoint(frame, (1200, 550), f"COG_vertical_vel: {COG_vertical_vel}", color=(255, 255, 255), draw_circle=False)
            # draw_keypoint(frame, (1200, 600), f"flight_vel: {flight_vel}", color=(255, 255, 255), draw_circle=False)
            # draw_keypoint(frame, (1200, 650), f"COG_cal_height: {COG_cal_height}", color=(255, 255, 255), draw_circle=False)

        # 更新 COG_height 當 COG 角度跨越 270 度時
        if (200 < COG_last_angle < 270 and COG_angle >= 270) or (COG_last_angle > 270 and 200 < COG_angle <= 270):
            COG_height = (500 - COG[1]) * 1.08  # 更新 COG_height 為當前 COG 的 y 值
        else: 
            COG_height = COG_last_height


    ###------右下角高度表示------___________________________________________________________________________________________________________________________________
    # 提取 CT 和 COG 的 y 座標
    CT_2dy = projected_points_2D['CT'][1]  # CT 的 y 座標 
    TL_2dy = projected_points_2D['TL'][1]  # COG 的 y 座標
    # 設置條件並更新 COG_flight_height 列表
    if CT_2dy < TL_2dy and 240 < COG_angle < 300 or CT_2dy < TL_2dy and Dis_COG > 150: 
        if len(COG_flight_height) == 0 or COG_cal_height != COG_flight_height[-1]:
            COG_flight_height.append(COG_cal_height)
            
        # 確保 COG_flight_height 的長度不超過 7
        if len(COG_flight_height) > 7:
            COG_flight_height.pop(0)

        # 更新 COG_flight_height_frames 列表，將 last_update_frame 加入其中
        if last_update_frame is not None:
            COG_flight_height_frames.append(last_update_frame)
        
        # 更新數據暫存表
        frame_data_table[frame_count] = {
            "COG_cal_height": COG_cal_height,
            "COG_angle": COG_angle,
            "COG_vertical_vel": COG_vertical_vel
        }

    # 創建右下角COG_flight_height 圖形圖像
    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)

    # 繪製 COG_flight_height 折線圖
    ax.plot(range(1, len(COG_flight_height) + 1), COG_flight_height, color='b', marker='o')  # 使用藍色線條並添加標記點
    ax.set_ylim(6000, 13000)  # 設定縱軸範圍
    ax.set_xlim(0, 8)         # 設定橫軸範圍

    # 設置縱軸標示
    ax.set_yticks(np.arange(7000, 13001, 2000))

    # 標題
    ax.set_title("Velocity Calculate COG Flight Height", fontsize=15)
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels([str(i) for i in range(1, 9)])
    ax.set_xlabel("Flips", fontsize=15)  # 設定 x 軸標籤
    ax.set_ylabel("Height", fontsize=15)  # 設定 y 軸標籤
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # 將圖像保存到內存中
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)  # 不要加額外的邊距
    plt.close(fig)
    buf.seek(0)

    # 讀取並嵌入到 OpenCV 圖像框架中
    plot_img = np.array(Image.open(buf))
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

    # 確保 plot_img 的大小為 (300, 700)
    plot_img_resized = cv2.resize(plot_img, (600, 300))  # 調整為 600x300 大小

    # 嵌入到原始影像的右下角
    frame[frame.shape[0]-300:frame.shape[0], frame.shape[1]-600:frame.shape[1]] = plot_img_resized

    
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
        'R_hand': (183, 239, 222),
        'COG': (0, 0, 220),
        }
    
    # 在图像上绘制每个点
    for name, (x, y) in projected_points_2D.items():
        color = colors_bgr.get(name, (0, 0, 0))  # 默认黑色
        cv2.circle(frame, (x, y), 5, color, -1)

    # 绘制连接线
    for start, end in connections:
        start_point = projected_points_2D[start]
        end_point = projected_points_2D[end]
        cv2.line(frame, start_point, end_point, (0, 0, 0), 2)  # 黑色线条，线宽 2


    ###-------繪製左下角滾動軸------_______________________________________________________________________________________________________________________
    COG_smoothed_history.append(abs(COG_tangential_vel))
    COG_angle_history.append(COG_angle)

    if len(COG_smoothed_history) > display_buffer:
        COG_smoothed_history.pop(0)
        COG_angle_history.pop(0)

    # 創建 Matplotlib 圖像
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    x_values = np.arange(len(COG_smoothed_history))
    ax.plot(x_values, COG_smoothed_history, color='r', label='COG')  # 繪製折線圖

    # 設定軸範圍
    ax.set_xlim(0, display_buffer)
    ax.set_ylim(0, 600)  # 縱軸最大值 600
    ax.set_yticks([150, 300, 450, 600])
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title("COG velocity", fontsize=15)
    ax.set_xticks([]) # 移除 x  軸的標籤


    # 判斷哪些點需要填充黃色背景
    yellow_regions = [((180 <= angle <= 240) or (angle >= 300)) for angle in COG_angle_history]
    yellow_regions = np.array(yellow_regions)
    ax.fill_between(x_values, 0, 600, where=yellow_regions,
                    facecolor=(1, 213/255, 99/255), alpha=0.5)
    
    # 計算歷史幀數列表
    history_frame_indices = list(range(frame_count - len(COG_smoothed_history) + 1, frame_count + 1))

    # 使用 COG_flight_height_frames，圈出要標記的數據點
    for idx, frame_idx in enumerate(history_frame_indices):
        if frame_idx in COG_flight_height_frames:
            y_pos = COG_smoothed_history[idx]
            # ax.plot(idx, y_pos, 'o', color='b', markersize=8)  # 用藍色圈圈標記

    # 在更新 COG_flight_height_frames 後，確保長度不超過某個值（例如 100）
    max_frames_length = 100
    if len(COG_flight_height_frames) > max_frames_length:
        COG_flight_height_frames = COG_flight_height_frames[-max_frames_length:]
        
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


    return frame, rot_origin,  rot_origin_proj, COG_angle, flight_vel, COG_height, COG_cal_height






