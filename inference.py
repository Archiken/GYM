import cv2
from ultralytics import YOLO
import time

start_time = time.time()
# 加载你训练好的两个模型
model_path_1 = '/app/runs/pose/train7/weights/best.pt'  # 第一个模型的路径
model_path_2 = '/app/runs/pose/train17/weights/best.pt'  # 第二个模型的路径
model_1 = YOLO(model_path_1)
model_2 = YOLO(model_path_2)

# 输入和输出视频路径
input_video_path = '/app/TANG_Chia-Hung.mp4'  # 输入视频的路径
output_video_path = '/app/inference_result/TANG_withcolor.mp4'  # 输出视频的路径

# 打开输入视频文件
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义视频编写器来保存输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编解码器
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 定义要连接的关键点索引列表，按顺序连接
keypoint_connections_body1 = [(0, 1), (1, 2)]  # 橘色
keypoint_connections_body2 = [(2, 3), (3, 4), (4, 5)]  # 黄色
keypoint_connections_body3 = [(1, 6), (6, 7), (7, 8)]  # 绿色
keypoint_connections_pole = [(0, 1), (1, 2), (2, 3), (3, 4)]  # 杆的连接

# 逐帧读取输入视频
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用两个 YOLOv8 模型分别进行姿势检测
    results_1 = model_1(frame)
    results_2 = model_2(frame)

    # 删除框框以确保不绘制框框
    for result in results_1:
        result.boxes = None  # 清空第一个模型检测到的框框信息
    for result in results_2:
        result.boxes = None  # 清空第二个模型检测到的框框信息

    # 绘制姿势关键点和连接线
    for result_1, result_2 in zip(results_1, results_2):
        # 获取第一个模型的关键点并绘制
        keypoints_1 = result_1.keypoints.xy.cpu().numpy().tolist()  # 转换为 Python 列表
        annotated_frame = result_1.plot()

        # 绘制第一个模型的关键点
        for idx, person_keypoints in enumerate(keypoints_1):
            for i, (x, y) in enumerate(person_keypoints):
                color = (255, 255, 255)  # 默认白色
                if i in [0, 1, 2]:
                    color = (0, 87, 288)  # 橘色
                elif i in [3, 4, 5]:
                    color = (0, 208, 243)  # 黄色
                elif i in [6, 7, 8]:
                    color = (0, 255, 168)  # 绿色
                
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, color, -1)

        # 绘制第一个模型的关键点连接
        for person_keypoints in keypoints_1:
            for start_idx, end_idx in keypoint_connections_body1:
                if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                    start_x, start_y = person_keypoints[start_idx][:2]
                    end_x, end_y = person_keypoints[end_idx][:2]
                    cv2.line(annotated_frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (26, 87, 228), 2)

            for start_idx, end_idx in keypoint_connections_body2:
                if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                    start_x, start_y = person_keypoints[start_idx][:2]
                    end_x, end_y = person_keypoints[end_idx][:2]
                    cv2.line(annotated_frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (18, 167, 143), 2)

            for start_idx, end_idx in keypoint_connections_body3:
                if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                    start_x, start_y = person_keypoints[start_idx][:2]
                    end_x, end_y = person_keypoints[end_idx][:2]
                    cv2.line(annotated_frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (134, 198, 168), 2)

        # 获取第二个模型的关键点并绘制
        keypoints_2 = result_2.keypoints.xy.cpu().numpy().tolist()  # 转换为 Python 列表

        # 绘制第二个模型的关键点和连接
        for person_keypoints in keypoints_2:
            # 绘制第二个模型的关键点
            for i, (x, y) in enumerate(person_keypoints):
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, (255, 255, 255), -1)  # 白色

            # 绘制第二个模型的关键点连接
            for start_idx, end_idx in keypoint_connections_pole:
                if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                    start_x, start_y = person_keypoints[start_idx][:2]
                    end_x, end_y = person_keypoints[end_idx][:2]
                    cv2.line(annotated_frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (188, 155, 102), 2)  # 蓝色

    # 将带有姿势关键点和连接线的帧写入输出视频
    out.write(annotated_frame)

# 释放视频捕获和写入对象
cap.release()
out.release()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序運行: {elapsed_time:.2f} 秒")
print(f"视频已保存至: {output_video_path}")
