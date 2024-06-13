import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
data = pd.read_csv('result.csv', header=None, names=['yolov8n', 'yolov8n-sparsified', 'yolov8n-gray'])

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制数据
ax.plot(data['yolov8n'], label='yolov8n')
ax.plot(data['yolov8n-sparsified'], label='yolov8n-sparsified')
ax.plot(data['yolov8n-gray'], label='yolov8n-gray')

# 设置标题和标签
ax.set_title('Performance Comparison')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Performance Score')

# 显示图例
ax.legend()

# 显示图形
plt.show()