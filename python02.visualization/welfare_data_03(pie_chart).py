import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = ['Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dataIn = './../dataIn/'
dataOut = './../dataOut/'

df = pd.read_csv(dataIn + '고객복지데이터셋Cleaned.csv')

# print("\n파이 그래프(Pie Chart)")
# marriage = df['결혼상태']
# print("\n데이터의 고유값:", marriage.unique())
# print("데이터의 고유값 개수:", marriage.nunique())
#
# # 빈도수 출력
# print("\n데이터의 빈도수:")
# # 결혼 상태 데이터 준비
# marriage_counts = marriage.value_counts()
# print(marriage_counts)
#
# # 파이 그래프 그리기
# plt.figure(figsize=(6,6))
# plt.pie(
#     marriage_counts,
#     labels=marriage_counts.index,
#     autopct='%1.2f%%',  # 퍼센트 표시
#     startangle=90,      # 12시 방향에서 시작
#     colors=['#ff9999','#66b3ff','#99ff99','#ffcc99']  # 원하는 색상 지정
# )
# plt.title("결혼 상태 비율")
# plt.savefig('e01.piechart_plot_01.png')
