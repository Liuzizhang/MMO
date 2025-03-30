# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as stats

# # 加载数据集
# data =  pd.read_csv('E:\BMSTU\一下\jqxx\heart.csv') 

# # 定义一个函数来绘制诊断图（直方图和Q-Q图）
# def diagnostic_plots(df, variable):
#     plt.figure(figsize=(15, 6))
    
#     # 直方图
#     plt.subplot(1, 2, 1)
#     df[variable].hist(bins=30, edgecolor='black', alpha=0.7)
#     plt.title(f'Histogram of {variable}')
    
#     # Q-Q图
#     plt.subplot(1, 2, 2)
#     stats.probplot(df[variable], dist="norm", plot=plt)
#     plt.title(f'Q-Q Plot of {variable}')
    
#     plt.show()

# # 对原始的trestbps列进行诊断
# diagnostic_plots(data, 'trestbps')

# # 应用Box-Cox变换
# data['trestbps_boxcox'], param = stats.boxcox(data['trestbps'])

# print(f'Optimal λ value for Box-Cox transformation: {param}')

# # 对变换后的price_boxcox列进行诊断
# diagnostic_plots(data, 'trestbps_boxcox')

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
# from sklearn.impute import SimpleImputer

# # 加载数据集
# data =  pd.read_csv('E:\BMSTU\一下\jqxx\waves_month_1.csv') 
# df = pd.read_csv('E:\BMSTU\一下\jqxx\waves_month_1.csv') 

# # 查看数据集的结构
# print(df.info())

# # 检查缺失值
# print(df.isnull().sum())

# # 填充缺失值
# imputer = SimpleImputer(strategy='median')
# numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
# df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# # 假设我们要预测的目标变量是 'Hs'，其他数值型列为特征
# X = df[numeric_columns].drop('Hs (m)', axis=1)
# y = df['Hs (m)']

# # 使用互信息方法选择5个最佳特征
# selector_mutual_info = SelectKBest(score_func=mutual_info_regression, k=5)
# X_new_mutual_info = selector_mutual_info.fit_transform(X, y)

# # 获取选中的特征名称
# selected_features_mutual_info = X.columns[selector_mutual_info.get_support()]

# print("\nSelected features using mutual information:")
# print(selected_features_mutual_info.tolist())

# # 可视化特征分数
# def plot_feature_scores(selector, title):
#     scores = selector.scores_
#     features = X.columns
#     plt.figure(figsize=(10, 6))
#     plt.barh(features, scores)
#     plt.xlabel('Feature Scores')
#     plt.title(title)
#     plt.gca().invert_yaxis() 

# plot_feature_scores(selector_mutual_info, 'Feature Scores using Mutual Information')
# plt.show()  
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('E:\BMSTU\一下\jqxx\heart.csv') 

data['chol'].hist(bins=30, edgecolor='black', alpha=0.7)

plt.title('Histogram of chol')
plt.xlabel('chol')
plt.ylabel('Frequency')

plt.show()
