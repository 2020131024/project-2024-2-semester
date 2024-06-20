import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/조진우/PycharmProjects/pythonProject/data3.csv', encoding='cp949')
df.info()

a = df['이름']
b = df['타율']
c = df['OPS']
d = df['WAR']
e = df['WRC+']

a_val = a.values
b_val = b.values
c_val = c.values
d_val = d.values
e_val = e.values

a_list = a_val.tolist()
b_list = b_val.tolist()
c_list = c_val.tolist()
d_list = d_val.tolist()
e_list = e_val.tolist()

dff = pd.DataFrame({'이름': a_list, '타율': b_list, 'OPS': c_list, 'WAR': d_list})
ax = dff.plot(x='이름', y=['타율', 'OPS', 'WAR'], marker='o')
ax.set_xlabel('이름')
ax.set_ylabel('Unit')
ax.legend(['타율', 'OPS', 'WAR'], loc='upper left')
ax.set_xticks(a_list)
ax.set_yticks([0, 10])
ax.set_title('타율과 OPS 및 WAR의 상관관계')

plt.show()

