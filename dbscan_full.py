import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 记录开始时间
start_time = time.time()

print("正在读取完整数据集...")
df = pd.read_csv('creditcard.csv')
print(f"数据形状: {df.shape}")
print(f"异常样本数量: {df['Class'].sum()} (占比: {df['Class'].mean()*100:.4f}%)")

# 数据预处理
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

# 标准化特征
print("正在标准化特征...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用之前的最佳参数
best_eps = 1.5
best_min_samples = 5
print(f"使用最佳参数: eps={best_eps}, min_samples={best_min_samples}")

# 训练DBSCAN模型
print("正在训练DBSCAN模型...")
model_start_time = time.time()
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels = dbscan.fit_predict(X_scaled)
model_end_time = time.time()
print(f"DBSCAN训练时间: {model_end_time - model_start_time:.2f}秒")

# 将噪声点(-1)标记为异常点(1)，其他点标记为正常点(0)
predictions = np.where(labels == -1, 1, 0)

# 评估结果
print("\n模型评估结果:")
print("分类报告:")
print(classification_report(y, predictions))

print("混淆矩阵:")
conf_matrix = confusion_matrix(y, predictions)
print(conf_matrix)

# 计算F1分数
f1 = f1_score(y, predictions)
print(f"F1分数: {f1:.4f}")

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['正常', '异常'], 
            yticklabels=['正常', '异常'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('DBSCAN混淆矩阵')
plt.savefig('dbscan_confusion_matrix.png')
plt.close()

# 可视化结果(使用PCA降维)
print("正在使用PCA降维进行可视化...")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
# 绘制正常交易点（蓝色）
plt.scatter(X_pca[predictions == 0, 0], X_pca[predictions == 0, 1], c='blue', s=5, alpha=0.1, label='正常交易')
# 绘制检测到的异常点（红色）
plt.scatter(X_pca[predictions == 1, 0], X_pca[predictions == 1, 1], c='red', s=20, alpha=0.5, label='检测到的异常')
# 绘制实际异常点（黄色边框）
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], facecolors='none', edgecolors='yellow', s=40, alpha=0.7, label='实际异常')

plt.title('DBSCAN聚类结果 (PCA降维)')
plt.legend()
plt.savefig('dbscan_full_results.png', dpi=300)
plt.close()

print(f"\n总运行时间: {time.time() - start_time:.2f}秒")
print("分析完成！") 