Motivation:
선형 회귀 분석(linear regression)은 주어진 데이터를 바탕으로 **독립 변수(X)**와 종속 변수(Y) 사이의 관계를 직선(선형 함수)으로 모델링하는 통계적 기법이다. 데이터가 선형적인 관계를 가진다고 가정할 때, 가장 적합한 직선을 찾아 새로운 데이터를 예측하거나 변수 간의 관계를 이해하는 데 사용할 수 있다. 이를 이용하여 한 학생의 4번의 시험점수를 토대로 5번 째 시험점수를 비교하는 코드를 pytorch를 활용해서 짜볼 수 있다.
또한 각 3명의 학생의 5번의 시험점수를 토대로 최종점수(final score)를 받은 상태에서 4번 째 학생의 5번의 시험점수를 토대로 4번 째 학생의 최종점수를 예측하는 model을 짜볼 수 있다. (multivariable linear regression)

Model:
pytorch를 Anaconda 환경을 활용한 pycharm을 이용해 사용하고 시각적 데이터를 위해 matplotlib를 사용한다.

Performance
선형 회귀 모델의 아이디어:
예측값 H(x)=W⋅x+b

H(x): 예측값 (Hypothesis)
W: 가중치 또는 기울기 (Weight)
x: 입력값 (독립 변수)
b: 절편 (Bias)
y: 실제값 (종속 변수, 타겟값)

(1) 주어진 데이터를 기반으로 H(x)를 계산한다.

(2) (Cost function(loss function))을 계산한다.
Cost(W,b)=(∑(H(x_i) - y_i)^2)/m

m: 데이터 샘플의 개수
H(x_i):i-번째 데이터의 예측값
y_i:i-번째 데이터의 실제값

우리의 Goal은 다음 Cost함수의 값을 최소로 줄이는 W와b 를 찾는것이다.

(3)경사하강법으로 W와b찾기 (gradient descent)

W:=W−α⋅(∂(Cost(W,b)/∂W)
b:=b−α⋅(∂(Cost(W,b)/∂b)

α: 학습률 (Learning Rate)

학습률 α는 W의 값을 변경할 때 얼마나 크게 변경할 지를 나타내는 값이다. 학습률이 너무커도 안좋으므로 적당한 학습률을 설정하는 것이 좋다.

이제 위의 방식을 토대로 선형회귀(linear regression) 및 다중선형회귀(multivariable linear regression)을 진행한다

1.선형회귀(linear regreesion)
한 학생의 4번의 시험점수를 토대로 5번 째 시험점수를 예측하는 선형 회귀 분석(linear regression)을 실행하고 시각화 그래프를 그린다.

DATA SET:학생의 4번의 시험점수 : 83, 91, 77, 85
학습률 : 0.01
실행횟수 : 2000
선형회귀모델을 사용해 나온 W,b,cost의 값과 예상되는 시험점수:
W:-0.712
b:85.741
cost:24.211256
점수:82.1347(82)
