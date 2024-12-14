import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# OpenMP 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.set_num_threads(1)

torch.manual_seed(1)

#데이터: 5개의 시험 점수와 최종 점수
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
