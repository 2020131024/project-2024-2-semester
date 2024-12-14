import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# OpenMP 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.set_num_threads(1)

torch.manual_seed(1)

#데이터
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
