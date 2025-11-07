import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

# 음수 부호 경고 메시지 없애기
plt.rcParams['axes.unicode_minus'] = False

dataIn, dataOut = './../dataIn/', './../dataOut/'

df = pd.read_csv(dataIn + 'auto-mpg.csv', header=None)
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
print(df.head())

print('상관 계수 확인')
corr = df.corr(numeric_only=True)
print(corr)

plt.figure(figsize=(10, 8))

# 상관 행렬의 상삼각 행렬 마스크 생성 (대칭이므로 절반만 보여줌)
# triu : Upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', cbar=True, square=True)

plt.title('변수들간 상관 계수', size=15)
filname = 'correlation.png'
plt.savefig(dataOut + filname)
print(f'{filname} 파일 생성됨')












