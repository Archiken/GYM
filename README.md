2024/09/28
YOLO v8自動生成的子資料夾中，第一層為runs，第二層為pose
第一次的訓練的所有結果會放入資料夾train中，第二次為train2；第三次為train3依序類推
目前上傳的兩個版本: 
train7為偵測人體骨架的模型
train18為偵測單槓四個點，以及緩衝墊兩個點的模型；用來建構3D空間的數據

一、環境建置
Docker image: https://hub.docker.com/repository/docker/archilin1/baseballpose/general

二、程式執行
1. analysis.py會抓取geometry_visualization.py並把分析影片生成。
2. inference.py則是單純用來檢視YOLO模型的成效，執行起來較快，但不會顯示分析結果。
