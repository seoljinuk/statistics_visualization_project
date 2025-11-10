import pandas as pd

print('\n시리즈 생성')
name = ['콜드브루', '에스프레소', '아메리카노', '카페라떼', '딸기라떼', '카페모카', '캐모마일티']
price = [5000, 6000, 4000, 8000, 7000, 3000, 3500]
beverage = pd.Series(data=price, index=name)

beverage.name = '커피 목록'
beverage.index.name = '품목명'

print(beverage)

print('\n# Series 주요 속성 출력')
print(f'values : {beverage.values}')
print(f'index : {beverage.index}')
print(f'name : {beverage.name}')
print(f'dtype : {beverage.dtype}')
print(f'size : {beverage.size}')
print(f'shape : {beverage.shape}')
print(f'ndim : {beverage.ndim}')
print(f'empty : {beverage.empty}')
print(f'axes : {beverage.axes}')
print(f'T : {beverage.T}')

print('\n# 값 관련 속성 출력')
print(f'결측값 포함 여부 : {beverage.hasnans}')
print(f'오름차순 여부 : {beverage.is_monotonic_increasing}')
print(f'내림차순 여부 : {beverage.is_monotonic_decreasing}')
print(f'중복 값 존재 여부 : {beverage.is_unique}')
print(f'총 바이트 크기 : {beverage.nbytes}')

print('\n# 인덱스 관련 속성 출력')
print(f'색인 이름 : {beverage.index.name}')
print(f'색인 실제 값 : {beverage.index.values}')
print(f'색인 자료형 : {beverage.index.dtype}')
print(f'색인 중복 여부 : {beverage.index.is_unique}')

# ──────────────────────────────
# 색인의 이름으로 값 읽기
# ──────────────────────────────
print("\n색인의 이름으로 값 읽기 : beverage[['카페라떼']]")
print(beverage[['카페라떼']])

# ──────────────────────────────
# 라벨 이름으로 슬라이싱
# ──────────────────────────────
print("\n라벨 이름으로 슬라이싱 : beverage['카페라떼':'카페모카']")
print(beverage['카페라떼':'카페모카'])

# ──────────────────────────────
# 여러 개의 색인 이름으로 데이터 읽기
# ──────────────────────────────
print("\n여러 개의 색인 이름으로 데이터 읽기 : beverage[['카페라떼', '캐모마일티']]")
print(beverage[['카페라떼', '캐모마일티']])

# ──────────────────────────────
# 정수를 이용한 데이터 읽기
# ──────────────────────────────
print("\n정수를 이용한 데이터 읽기 : beverage.iloc[[2]]")
print(beverage.iloc[[2]])

# ──────────────────────────────
# 0, 2, 4번째 데이터 읽기
# ──────────────────────────────
print("\n0, 2, 4번째 데이터 읽기 : beverage[0:5:2]")
print(beverage[0:5:2])

# ──────────────────────────────
# 1, 3, 5번째 데이터 읽기
# ──────────────────────────────
print("\n1, 3, 5번째 데이터 읽기 : beverage.iloc[[1, 3, 5]]")
print(beverage.iloc[[1, 3, 5]])

# ──────────────────────────────
# 슬라이싱 사용하기
# ──────────────────────────────
print("\n슬라이싱 사용하기 : beverage[3:6]  # from <= 결과 < to")
print(beverage[3:6])

# ──────────────────────────────
# 2번째 항목의 값 변경
# ──────────────────────────────
print("\n2번째 항목의 값 변경 : beverage.iloc[2] = 4500")
beverage.iloc[2] = 4500
print(beverage)

# ──────────────────────────────
# 2번째부터 4번째 까지 항목의 값 변경
# ──────────────────────────────
print("\n2번째부터 4번째 까지 항목의 값 변경 : beverage[2:5] = 3200")
beverage[2:5] = 3200
print(beverage)

# ──────────────────────────────
# 콜드브루과 카페라떼만 5500로 변경
# ──────────────────────────────
print("\n콜드브루과 카페라떼만 5500로 변경 : beverage[['콜드브루', '카페라떼']] = 5500")
beverage[['콜드브루', '카페라떼']] = 5500
print(beverage)

# ──────────────────────────────
# 짝수 행만 7000로 변경
# ──────────────────────────────
print("\n짝수 행만 7000로 변경 : beverage[0::2] = 7000")
beverage[0::2] = 7000
print(beverage)

print("\n유일한 값, 값 세기, 멤버십")
# ──────────────────────────────
# unique() : 시리즈에서 중복 제거 후 고유값 배열 반환
# ──────────────────────────────
print(f'\n고유값(unique) : {beverage.unique()}')

# ──────────────────────────────
# value_counts() : 각 값이 몇 번 나오는지 빈도수 출력
# ──────────────────────────────
print(f'\n값별 빈도수(value_counts) :\n{beverage.value_counts()}')

# ──────────────────────────────
# isin() : 특정 값이 시리즈에 포함되어 있는지 불리언 시리즈로 반환
# ──────────────────────────────
check_values = [3000, 6000, 7000]
print(f'{check_values} 포함 여부(isin) :\n{beverage.isin(check_values)}')

# isin을 활용하여 실제 값 출력
selected = beverage[beverage.isin(check_values)]
print(f'{check_values} 포함 값 출력 :\n{selected}')

print('\n시리즈 내용 확인')
print(beverage)


print('\n공용 산술 연산 메소드(스칼라)')
scalar = 500

print("덧셈(add) 결과 :")
print(beverage.add(scalar))

print("\n뺄셈(sub) 결과 :")
print(beverage.sub(scalar))

print("\n곱셈(mul) 결과 :")
print(beverage.mul(scalar))

print("\n나눗셈(div) 결과 :")
print(beverage.div(scalar))

print('\n공용 산술 연산 메소드(시리즈끼리 연산)')
other_price = pd.Series([1000, 500, 300, 200, 400, 600, 700],
                        index=['콜드브루', '에스프레소', '아메리카노', '카페라떼', '딸기라떼', '카페모카', '캐모마일티'])

print("덧셈(add) 결과 :")
print(beverage.add(other_price))

print("\n뺄셈(sub) 결과 :")
print(beverage.sub(other_price))

print("\n곱셈(mul) 결과 :")
print(beverage.mul(other_price))

print("\n나눗셈(div) 결과 :")
print(beverage.div(other_price))

print('\n관계 연산 관련 메소드')
# 기준 값 설정
threshold = 5500

print(f'\n{threshold}보다 큰 값(gt) 여부 :\n{beverage.gt(threshold)}\n')

print(f'\n{threshold} 이상(ge) 여부 :\n{beverage.ge(threshold)}\n')

print(f'\n{threshold}보다 큰 값(gt) :\n{beverage[beverage.gt(threshold)]}\n')
print(f'\n{threshold} 이상(ge) 값 :\n{beverage[beverage.ge(threshold)]}\n')
print(f'\n{threshold}보다 작은 값(lt) :\n{beverage[beverage.lt(threshold)]}\n')
print(f'\n{threshold} 이하(le) 값 :\n{beverage[beverage.le(threshold)]}\n')
print(f'\n{threshold}과 같은 값(eq) :\n{beverage[beverage.eq(threshold)]}\n')
print(f'\n{threshold}와 다른 값(ne) :\n{beverage[beverage.ne(threshold)]}')