Motivation:
선형 회귀 분석(linear regression)은 주어진 데이터를 바탕으로 **독립 변수(X)**와 종속 변수(Y) 사이의 관계를 직선(선형 함수)으로 모델링하는 통계적 기법이다.

데이터가 선형적인 관계를 가진다고 가정할 때, 가장 적합한 직선을 찾아 새로운 데이터를 예측하거나 변수 간의 관계를 이해하는 데 사용할 수 있다.

이를 이용하여 한 학생의 4번의 시험점수를 토대로 5번 째 시험점수를 비교하는 코드를 pytorch를 활용해서 짜볼 수 있다.

또한 각 3명의 학생의 5번의 시험점수를 토대로 최종점수(final score)를 받은 상태에서 4번 째 학생의 5번의 시험점수를 토대로 4번 째 학생의 최종점수를 예측하는 model을 짜볼 수 있다. (multivariable linear regression)

Model:

pytorch를 Anaconda 환경을 활용한 pycharm을 이용해 사용하고 시각적 데이터를 위해 matplotlib를 사용한다.

Performance

1)선형 회귀 모델의 아이디어:

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

(3) 경사하강법으로 W와b찾기 (gradient descent)

W:=W−α⋅(∂(Cost(W,b)/∂W)

b:=b−α⋅(∂(Cost(W,b)/∂b)

α: 학습률 (Learning Rate)

학습률 α는 W의 값을 변경할 때 얼마나 크게 변경할 지를 나타내는 값이다. 학습률이 너무커도 안좋으므로 적당한 학습률을 설정하는 것이 좋다.

2)다중 선형 회귀 모델의 아이디어:

예측값 H(X)=X⋅W+b

X: 독립 변수(Feature) 행렬 (m×n, m은 데이터 샘플 개수, n은 Feature 개수)

W: 가중치(Weight) 벡터 (n×1, 각 Feature에 대한 가중치)

b: 절편(Bias) (1×1)

Y: 실제값(Target) 벡터 (m×1)

(1) 주어진 데이터를 기반으로 H(X)를 계산한다.

(2) (Cost function(loss function))을 계산한다.

Cost(W,b)=(∑(H(x_i) - y_i)^2)/m

m: 데이터 샘플의 개수

H(x_i):i-번째 데이터의 예측값

y_i:i-번째 데이터의 실제값

(3) 경사하강법으로 W와b찾기 (gradient descent)

W:=W−α⋅(∂(Cost(W,b)/∂W)

b:=b−α⋅(∂(Cost(W,b)/∂b)

α: 학습률 (Learning Rate)

이제 위의 방식을 토대로 선형회귀(linear regression) 및 다중선형회귀(multivariable linear regression)을 진행한다

1.선형회귀(linear regreesion):

한 학생의 4번의 시험점수를 토대로 5번 째 시험점수를 예측하는 선형 회귀 분석(linear regression)을 실행하고 시각화 그래프를 그린다.

DATA SET:학생의 4번의 시험점수 : 83, 91, 77, 85

학습률 : 0.01

실행횟수 : 2000

선형회귀모델을 사용해 나온 W,b,cost의 값과 예상되는 시험점수:

W:-0.712

b:85.741

cost:24.211256

시험점수:82.1347(82)

다음은 위 선형회귀모델을 사용해 나온 예측값을 matplotlib를 활용해서 시각화한 것과 pycharm에서 사용한 코드다.

![선형회귀사진1](https://github.com/user-attachments/assets/a8457912-19a9-4aee-8ce6-b925e2cd1fb6)
```
# torch, matplotlib import
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# OpenMP 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.set_num_threads(1)

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3], [4]])
y_train = torch.FloatTensor([[83], [91], [77], [85]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Optimizer
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999
for epoch in range(nb_epochs + 1):
    # H(x)
    hypothesis = x_train * W + b

    # cost
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번째마다 print
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} W: {W.item():.3f}, b: {b.item():.3f}, Cost: {cost.item():.6f}')

# 5번째 시험결과 예측
with torch.no_grad():
    prediction = W * 5 + b
    print(f"Predicted score for exam 5: {prediction.item():.2f}")

# 그래프
plt.scatter(x_train.numpy(), y_train.numpy(), label='Train Data', color='blue')
x_line = torch.linspace(1, 5, 100).view(-1, 1)
y_line = W.detach() * x_line + b.detach()
plt.plot(x_line.numpy(), y_line.numpy(), label='Regression Line', color='red')
plt.title("Linear Regression")
plt.xlabel("Exam Number")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()
```

2.다중선형회귀(multivariable linear regression):

3명의 학생의 5번의 시험점수와 그것을 토대로 최종점수가 나와있다.

4번 째 학생의 5번의 시험점수가 주어졌을 때 4번 째 학생의 최종점수를 예측하는 다중 선형 회귀 분석(multivariable linear regression)을 실행하고 시각화 그래프를 그린다.

최종점수 계산방식 : 각 시험점수를 a b c d e 라고하면 (a+b+c)*0.2 + (d+e)* 0.3

위 최종점수 계산방식에 따른 4번 째 학생의 실제 최종점수와 모델을 이용한 최종점수를 비교해보자.

DATA SET:3명의 학생의 5번의 시험점수와 최종점수

학생 1: 80 88 92 75 81 , 99

학생 2: 95 82 93 71 70 , 96

학생 3: 83 91 77 85 82 , 100

학생 4: 90 84 92 79 81 , ?(실제점수는 101.2)

학습률 : 1e-5

실행횟수 : 999

다중선형회귀모델을 사용해 나온 W,b,cost값과 예상되는 최종점수:

W: [[0.18459107]

 [0.2768171 ]
 
 [0.20399831]
 
 [0.25735104]
 
 [0.26705465]]
 
b:0.00

cost: 0.026576

최종점수:100.60

실제 점수와 0.6점 차이남을 알 수 있다.

다음은 위 다중선형회귀모델을 사용해 나온 예측값과 실제값을 matplotlib를 활용해서 시각화한 것과 pycharm에서 사용한 코드다.

![다중선형회귀사진](https://github.com/user-attachments/assets/3ddb2fd6-5c63-497f-9bc5-ba33724ba731)

```
# torch, matplotlib import
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# OpenMP 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.set_num_threads(1)

torch.manual_seed(1)

# 데이터: 5개의 시험 점수와 최종 점수
x_train = torch.FloatTensor([[80, 88, 92, 75, 81],
                             [95, 82, 93, 71, 70],
                             [83, 91, 77, 85, 82]])
y_train = torch.FloatTensor([[99], [96], [100]])

W = torch.zeros((5, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Optimizer
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 999
for epoch in range(nb_epochs + 1):
    # H(x)
    hypothesis = x_train.matmul(W) + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    # Cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 print
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

# 4번째 학생의 최종점수 예측
with torch.no_grad():
    predictions = x_train.matmul(W) + b
    new_input = torch.FloatTensor([[90, 84, 92, 79, 81]])
    new_prediction = new_input.matmul(W) + b
    print(f"Predicted score for new input {new_input.squeeze().tolist()}: {new_prediction.item():.2f}")

    print(f"Final Weights (W): \n{W.numpy()}")
    print(f"Final Bias (b): {b.item():.2f}")
    print(f"Predicted score for new input {new_input.squeeze().tolist()}: {new_prediction.item():.2f}")
  
# 그래프
plt.figure(figsize=(8, 6))

# 실제값 vs 예측값
plt.scatter(range(len(y_train)), y_train.numpy(), label='Actual Values', color='blue')
plt.scatter(range(len(predictions)), predictions.numpy(), label='Predicted Values', color='red')


plt.scatter([len(y_train)], [new_prediction.item()], label='New Prediction', color='green', marker='X', s=100)

# 그래프
plt.title("Actual vs Predicted Values in Multiple Linear Regression")
plt.xlabel("Sample Index")
plt.ylabel("Final Score")
plt.legend()
plt.grid()
plt.show()
```
