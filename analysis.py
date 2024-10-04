import cv2
import numpy as np
from ultralytics import YOLO
from geometry_visualization import draw_on_frame  # 從繪製模組導入函數

def process_video(input_video_path , output_type="video", output_path="output.mp4", frame_number=None):
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

    frame_count = 0

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
        frame = draw_on_frame(frame, results_1, results_2, frame_count)

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
# process_video(input_video_path= '/app/TANG_Chia-Hung.mp4', output_type="image", output_path="/app/inference_result/frame_175_analysis.jpg", frame_number=175)
process_video(input_video_path= '/app/TANG_Chia-Hung.mp4', output_type="video", output_path="/app/inference_result/Tang_output_3drot2.mp4")
