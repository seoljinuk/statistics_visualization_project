import numpy as np 
import pandas as pd

coffee_name = ['콜드브루', '에스프레소', '아메리카노', '카페라떼', '딸기라떼']
coffee_brand = ['스타벅스', '할리스', '이디야', '투썸', '커피빈']
imsi_data = list(10 * onedata for onedata in range(1, 26))
print(imsi_data)

sale_data = pd.DataFrame(np.reshape(imsi_data, (5, 5)),
                     index=coffee_name,
                     columns = coffee_brand  )

sale_data.columns.name = '커피유형'
sale_data.index.name = '브랜드명'

print(f'열 색인 이름 : {sale_data.columns.name}')
print(f'행 색인 이름 : {sale_data.index.name}')
print(f'열 색인 정보 : {sale_data.columns}')
print(f'행 색인 정보 : {sale_data.index}')
print(f'각 열의 데이터 타입(dtypes) :\n{sale_data.dtypes}')
print(f'데이터 값(values) :\n{sale_data.values}')
print(f'전치(T) :\n{sale_data.T}')
print(f'요소 총 개수(size) : {sale_data.size}')
print(f'데이터 형태(shape) : {sale_data.shape}')
print(f'차원 수(ndim) : {sale_data.ndim}')
print(f'첫 번째 열을 리스트로 변환(toList) : {sale_data["스타벅스"].tolist()}')
print(f'메모리 사용량(memory_usage) :\n{sale_data.memory_usage()}')

print(sale_data)


print("\n1번째 행을 Series로 읽기 (iloc[1])")
print(sale_data.iloc[1])

print("\n1번째와 3번째 행 읽기 (iloc[[1,3]])")
print(sale_data.iloc[[1,3]])

print("\n짝수 행만 읽기 (iloc[0::2])")
print(sale_data.iloc[0::2])

print("\n홀수 행만 읽기 (iloc[1::2])")
print(sale_data.iloc[1::2])

print("\n'아메리카노' 행 읽기 (loc)")
print(sale_data.loc['아메리카노'])

print("\n'콜드브루'와 '카페라떼' 행 읽기 (loc[['콜드브루', '카페라떼']])")
print(sale_data.loc[['콜드브루', '카페라떼']])

print("\n람다 조건을 사용하여 '이디야' >= 130인 행만 추출")
print(sale_data.loc[lambda df: df['이디야'] >= 130])

print("\n복원 추출")
mytarget = np.random.choice(sale_data.index, 3)
print(mytarget)

result = sale_data.loc[ mytarget ]
print(result)

print('\n# 아메리카노의  이디야 실적 정보 가져 오기')
result = sale_data.loc[['아메리카노'], ['이디야']] # DataFrame
print(result)
 
print('\n# 딸기라떼와 카페라떼의 이디야/투썸 정보 가져 오기')
result = sale_data.loc[['딸기라떼', '아메리카노'], ['이디야', '투썸']]
print(result)

print('\n# 연속적인 데이터 가져 오기')
result = sale_data.loc['에스프레소':'카페라떼', '이디야':'투썸']
print(result)
 
print('\n# 에스프레소~카페라떼까지  할리스 실적 정보 가져 오기')
result = sale_data.loc['에스프레소':'카페라떼', ['할리스']]
print(result)
 
print('\nBoolean으로 데이터 처리하기')
result = sale_data.loc[[False, True, True, False, True]]
print(result)
 
print('\n할리스 실적이 100이하인 항목들')
result = sale_data.loc[ sale_data['할리스'] <= 100 ]
print(result)
 
print('\n투썸 실적이 140인 항목들')
result = sale_data.loc[ sale_data['투썸'] == 140 ]
print(result)

cond1 = sale_data['할리스'] >= 70
cond2 = sale_data['투썸'] >= 140

print(type(cond1))
print('-' * 40)

print(cond1)
print('-' * 40)

print(cond2)
print('-' * 40)
 
df = pd.DataFrame([cond1, cond2])
print(df)
print('-' * 40)

print(df.all())
print('-' * 40)

print(df.any())
print('-' * 40)

result = sale_data.loc[ df.all() ]
print(result)
print('-' * 40)

result = sale_data.loc[ df.any() ]
print(result)
print('-' * 40)
  
print('\n람다 함수의 사용')
result = sale_data.loc[ lambda df : df['이디야'] >= 130  ]
print(result)

print('\n파생열 생성(스타벅스 + 할리스)')
print('# 기존 열에 대한 연산을 이용하여 신규 열을 생성하는 방법')
sale_data['총합'] = (sale_data['스타벅스'] + sale_data['할리스'])
print(sale_data)

print('# 특정 scalar을 사용하여 신규 열을 생성하는 방법')
sale_data['스타벅스100'] = (sale_data['스타벅스'] + 100)
print(sale_data)

print('\n컬럼 순서 재배치하기')
major_column = ['총합', '투썸', '스타벅스', '스타벅스100']
minor_column = ['할리스', '이디야',  '커피빈']

myordering = major_column + minor_column
print('컬럼 순서 재배치 현황 : \n' + str(myordering))

result = set(sale_data.columns) == set(myordering)
print('최초 컬럼 정보와 동일한가 ? ' + str(result))

sale_data_new = sale_data[myordering]
print(sale_data_new)

print('\n행/열색인 삭제하기')
REMOVED_ROW, REMOVED_COLUMN = '카페라떼', '스타벅스100'
sale_data_removed_row = sale_data.drop(labels=REMOVED_ROW)
sale_data_removed_column = sale_data.drop(labels=REMOVED_COLUMN, axis=1)

print("\n원본 데이터프레임:")
print(sale_data)

print(f"\n제거된 행 : {REMOVED_ROW}")
print(sale_data_removed_row)

print(f"\n제거된 열:{REMOVED_COLUMN}")
print(sale_data_removed_column)

sale_data.drop(labels=['총합', '스타벅스100'], inplace=True, axis=1)

print("\n원본 데이터프레임:")
print(sale_data)

print("\n데이터프레임 요약 정보:")
print(sale_data.info())

  
print('\n# 콜드브루과 아메리카노의 할리스 실적을 30으로 변경하기')
sale_data.loc[['콜드브루', '아메리카노'], ['할리스']] = 30
 
print('\n## 에스프레소부터 카페라떼까지 커피빈 실적을 80으로 변경하시오.')
sale_data.loc['에스프레소':'카페라떼', ['커피빈']] = 80
  
print('\n# 딸기라떼의 모든 실적을 50으로 변경하기')
sale_data.loc[['딸기라떼'], :] = 50
  
print('\n# 모든 사람의 이디야 컬럼을 60으로 변경하기')
sale_data.loc[:, ['이디야']] = 60
  
print('\n# 커피빈 실적이 150이하인 데이터를 모두 0으로 변경하기')
sale_data.loc[sale_data['커피빈'] <= 150 , ['커피빈', '이디야']] = 80
 
print('\n# 데이터 프레임 확인')
print(sale_data)



