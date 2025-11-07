import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

x = [[1, 2, 3, 4, 5], [30, 25, 20, 15, 10], [2005, 2010, 2015, 2020, 2025]]
y = [2, 3, 5, 7, 11]

# fig는 그림(Figure) 객체, axs는 Axes 객체(0 base)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

xlabels = ['방개수', '위치(도보 시간)', '신축 년도']

for idx in range(len(xlabels)):
    axs[idx].plot(x[idx], y, marker='o')
    axs[idx].set_title(xlabels[idx] + '와 가격', size=15)
    axs[idx].set_xlabel(xlabels[idx], size=12)
    axs[idx].set_ylabel('가격')

plt.ylim([0, 12])
plt.tight_layout() # 서브 플롯간의 간격 조율

dataOut = './../dataOut/'
filename = dataOut + 'house_value.png'
plt.savefig(filename)
print(filename + ' 파일이 저장되었습니다.')