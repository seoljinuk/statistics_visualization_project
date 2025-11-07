import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

cases = ['케이스01', '케이스02', '케이스03']

xlabels = ['대식구', '출퇴근시간', '깔끔한집']

mylabels = ['방의 개수', '위치', '건축 년도']
data = [(100, 60, 30), (30, 100, 60), (60, 30, 100)]

# 각 데이터에서 가장 큰 값의 인덱스를 찾아 그에 해당하는 라벨 이름을 리스트로 저장
titleName = [mylabels[item.index(max(item))] for item in data]

# 1x3 배열로 된 subplot 생성 (한 줄에 3개의 그래프)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 가로 12인치, 세로 4인치 크기의 그래프 배열 생성

# 각 데이터 케이스에 대한 파이 차트를 그리는 반복문
for i, row in enumerate(data):
    axes[i].pie(row, labels=mylabels, autopct='%1.1f%%', startangle=90)  # 각 케이스에 대한 파이 차트 생성
    axes[i].set_title('주요 요소 : ' + titleName[i], size=15)  # 가장 큰 값에 해당하는 요소를 제목으로 설정
    axes[i].set_xlabel(xlabels[i])  # 각 케이스에 해당하는 설명을 x축 레이블로 설정

dataOut = './../dataOut/'
filename = dataOut + 'houseSelect.png'
plt.savefig(filename)
print(filename + ' 파일이 저장되었습니다.')