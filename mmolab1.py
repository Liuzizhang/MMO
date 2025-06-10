import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('E:\BMSTU\一下\jqxx\heart.csv')  # 用你的数据集路径替换这里

# trestbps和age的相关性散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['trestbps'], alpha=0.6)
plt.title('trestbps vs. age Correlation')
plt.xlabel('age(years)')
plt.ylabel('trestbps')
plt.grid(True)
plt.show()

# target占比饼状图
outcome_counts = df['target'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(outcome_counts, labels=['target (0)', 'target (1)'], autopct='%1.1f%%', startangle=140, colors=['skyblue', 'salmon'])
plt.title('target')
plt.show()

# resting blood pressure 的分布箱线图
plt.figure(figsize=(10, 6))
plt.boxplot(df['trestbps'].dropna())
plt.title('resting blood pressure  Distribution')
plt.xticks([1], ['trestbps'])
plt.ylabel('target')
plt.grid(True)
plt.show()

# serum cholestoral in mg/dl和target的直方图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['chol'], bins=30, color='skyblue', edgecolor='black')
plt.title('Skin Thickness Distribution')
plt.xlabel('chol')
plt.ylabel('chol')
plt.subplot(1, 2, 2)
plt.hist(df['target'], bins=30, color='salmon', edgecolor='black')
plt.title('target')
plt.xlabel('target')
plt.ylabel('chol')
plt.tight_layout()
plt.show()


columns_of_interest = ['trestbps', 'age', 'chol']
corr = df[columns_of_interest].corr().values
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(columns_of_interest)), columns_of_interest, rotation=45)
plt.yticks(range(len(columns_of_interest)), columns_of_interest)
plt.title('Correlation Heatmap')
for i in range(len(corr)):
    for j in range(len(corr)):
        text = plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="w")
plt.show()
