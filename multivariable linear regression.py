import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# OpenMP 충돌 방지 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.set_num_threads(1)  # PyTorch에서 사용하는 스레드 수 제한

torch.manual_seed(1)

# 학습 데이터: 5개의 시험 점수와 최종 점수
x_train = torch.FloatTensor([[80, 88, 92, 75, 81],
                             [95, 82, 93, 71, 70],
                             [83, 91, 77, 85, 82]])
y_train = torch.FloatTensor([[99], [96], [100]])

# 모델 초기화
W = torch.zeros((5, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

# 학습
nb_epochs = 999
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = x_train.matmul(W) + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    # Cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 로그 출력
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

# 학습 데이터 예측
with torch.no_grad():
    predictions = x_train.matmul(W) + b  # 학습 데이터에 대한 예측
    new_input = torch.FloatTensor([[90, 84, 92, 79, 81]])  # 새로운 데이터
    new_prediction = new_input.matmul(W) + b
    print(f"Predicted score for new input {new_input.squeeze().tolist()}: {new_prediction.item():.2f}")

    print(f"Final Weights (W): \n{W.numpy()}")
    print(f"Final Bias (b): {b.item():.2f}")
    print(f"Predicted score for new input {new_input.squeeze().tolist()}: {new_prediction.item():.2f}")
# 그래프 시각화
plt.figure(figsize=(8, 6))

# 실제값 vs 예측값
plt.scatter(range(len(y_train)), y_train.numpy(), label='Actual Values', color='blue')  # 실제 값
plt.scatter(range(len(predictions)), predictions.numpy(), label='Predicted Values', color='red')  # 예측 값

# 새로운 데이터 예측 표시
plt.scatter([len(y_train)], [new_prediction.item()], label='New Prediction', color='green', marker='X', s=100)

# 그래프 설정
plt.title("Actual vs Predicted Values in Multiple Linear Regression")
plt.xlabel("Sample Index")
plt.ylabel("Final Score")
plt.legend()
plt.grid()
plt.show()
