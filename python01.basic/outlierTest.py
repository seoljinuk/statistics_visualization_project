import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42) # 랜덤 시드 배정

# 평균 50, 표준 편차 15를 갖는 100개의 랜덤 데이터 생성
data = np.random.normal(loc=50, scale=15, size=100)
print(data)

# 이상치 추가
data[95] = 150
data[96] = 160

# sorted는 정렬을 수행해주는 파이썬 내장 함수
mylist = [float(item) for item in sorted(data)]
print(mylist)

df = pd.DataFrame({'value':data})
print(df)

plt.rc('font', family='Malgun Gothic')

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['value'])
plt.title('이상치 데이터 보기')
plt.savefig('outlier.png')