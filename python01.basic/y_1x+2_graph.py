mydata = [(0, 2), (1, 3), (2, 4), (3, 5)]

x_data = [data[0] for data in mydata]
y_data = [data[1] for data in mydata]

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

plt.plot(x_data, y_data, marker='o', linewidth=1, label='직선 그래프')

plt.grid(True, which='both', axis='both', linestyle='--', linewidth=1)

plt.xlim([-0.5, 3.5])
plt.ylim([1.5, 5.5])
plt.title('Line Graph')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.legend()

dataOut = './../dataOut/'
filename = dataOut + 'y_1x+2_graph.png'

plt.savefig(filename)

print(filename + ' 파일이 저장되었습니다.')
