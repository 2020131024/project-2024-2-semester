from itertools import combinations

#임의의 리스트의 부분집합을 구하는 itertools 사용
def sub_lists(my_list):
    subs = []

    for i in range(0, len(my_list) + 1):

        temp = [list(x) for x in combinations(my_list, i)]

        if len(temp) > 0:
            subs.extend(temp)

    return subs

#input data로 정수 리스트 생성하기 ex)1 3 2 > [1, 3, 2]
a = list(map(int, input('입력:').split()))

#받은 리스트를 오름차순으로 정렬 ex)[1, 3, 2] > [1, 2, 3]
arr = sorted(a)

#n개의 정수를 받으므로 리스트 길이를 활용해 n 설정
n = len(arr)

#오름차순으로 되어있는 리스트의 부분집합 구하기 ex)[1, 2, 3] > [[], [1], [2], [3]. [1.2],..., [1,2,3]]
b = sub_lists(arr)
#for문을 활용하여 부분집합 중 [] 원소 제거
All = []
for i in range(len(b)):
    if len(b[i]) >= 1:
        All.append(list(b[i]))

#필요시 중간점검
#print(All)

#코드 목적에 부합하는 부분집합을 구하기 위해 for문을 활용해 각 list의 원소의 합이 n의 배수인치 확인하고 분류
multiple_n = []
for i in range((2**n)-1):
    if sum(All[i]) % n == 0:
        multiple_n.append(list(All[i]))

#필요시 중간점검
#print(multiple_n)

#원하는 조건을 만족하는 리스트들을 각 리스트의 길이별로 분류해서 출력
for i in range(len(multiple_n)):
    for j in range(1,n+1):
        if len(multiple_n[i]) == j:
            print(f'길이가 {j}인 리스트는 {multiple_n[i]}')