import cv2
import numpy as np
from ultralytics import YOLO
from geometry_visualization import draw_on_frame  # 從繪製模組導入函數
import csv
import time

start_time = time.time()

def process_video(input_video_path , output_type="video", output_path="output.mp4", frame_number=None):
    # 定義全局的數據暫存表
    frame_data_table = {}
    # 定義列表來記錄 COG_flight_height 更新的幀索引
    COG_flight_height_frames = []
    # 加载模型
    model_path_1 = '/app/runs/pose/train7/weights/best.pt'
    model_path_2 = '/app/runs/pose/train18/weights/best.pt'
    model_1 = YOLO(model_path_1)
    model_2 = YOLO(model_path_2)

    # 打开输入视频文件
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # 获取视频的基本信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 输出模式处理
    if output_type == "video":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # with open('rot_origin_output25.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Frame', 'rot_velocity', 'smoothed_velocity_3', 'smoothed_velocity_5'])

    frame_count = 0
    rot_origin_last = None  # 初始化上一幀的 rot_origin
    rot_proj_last = None
    # R_hip_last_angle = None
    # R_ankle_last_angle = None
    COG_last_angle = None
    flight_last_vel = None
    COG_last_height = None
    COG_last_cal_height = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取结束

        if frame_number is not None and frame_count != frame_number:
            frame_count += 1
            continue  # 跳到指定帧

        # 使用两个模型检测关键点
        results_1 = model_1(frame)
        results_2 = model_2(frame)

        # 使用分析繪製模組進行繪製
        
        #hip/ ankle版本
        # frame, rot_origin, rot_origin_proj, R_hip_angle, hip_smoothed, R_ankle_last_angle, ankle_smoothed = draw_on_frame(frame, results_1, 
        # results_2, frame_count, rot_origin_last, rot_proj_last, R_hip_last_angle, R_ankle_last_angle) 
        
        # R_hip_last_angle = R_hip_angle #更新後下次傳回去用
        # R_ankle_last_angle = R_ankle_last_angle

        # frame = draw_on_frame_with_plot(frame, hip_smoothed, ankle_smoothed)

        #COG 版本
        frame, rot_origin, rot_origin_proj, COG_angle, flight_vel, COG_height, COG_cal_height = draw_on_frame(
    frame, results_1, results_2, frame_count, rot_origin_last, rot_proj_last,
    COG_last_angle, flight_last_vel, COG_last_height, COG_last_cal_height,
    frame_data_table, COG_flight_height_frames)
        #COG 版本
        COG_last_angle = COG_angle
        flight_last_vel = flight_vel
        COG_last_height = COG_height
        COG_last_cal_height = COG_cal_height

        rot_origin_last = rot_origin #更新後下次傳回去用
        rot_proj_last = rot_origin_proj #更新後下次傳回去用

        # 打開 CSV 文件並以追加模式寫入新數據
        # with open('rot_speed.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     # 寫入當前幀數的...
        #     writer.writerow([frame_count, hip_ang_velocity, hip_smoothed])

        if output_type == "image":
            cv2.imwrite(output_path, frame)
            print(f"Image saved to {output_path}")
            break
        elif output_type == "video":
            out.write(frame)

        frame_count += 1
        print(f"Processed frame {frame_count}")

    cap.release()
    if output_type == "video":
        out.release()
        print(f"Video saved to {output_path}")

# 示例调用
# process_video(input_video_path= '/app/TANG_Chia-Hung.mp4', output_type="image", output_path="/app/inference_result/frame_534_analysis.jpg", frame_number=534)
process_video(input_video_path= '/app/TANG_Chia-Hung.mp4', output_type="video", output_path="/app/inference_result/TANG_Chia-Hung_verticalheight.mp4")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序運行: {elapsed_time:.2f} 秒")