x1, y1 = (0, 2)
x2, y2 = (1, 3)

# y = a*x + b
a = (y2 - y1) / (x2 - x1)
b = y1 - a * x1

print(f'직선의 기울기 : {a}')
print(f'y 절편 : {b}')

import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
# 음수 부호 경고 메시지 없애기
plt.rcParams['axes.unicode_minus'] = False

x_values = np.linspace(-1, 2, 100)
y_values = a * x_values + b

plt.figure(figsize=(6, 4))
plt.plot(x_values, y_values)
plt.scatter([x1, x2], [y1, y2], color='red', label='주어진 점')

# 그래프 꾸미기
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('두 점을 통과하는 직선')

filename = 'linearExpression.png'
dataOut = './../dataOut/'
plt.savefig(dataOut + filename)
print(f'{filename} 파일 저장')




