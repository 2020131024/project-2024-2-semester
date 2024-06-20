import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('C:/Users/조진우/PycharmProjects/pythonProject/data3.csv', encoding='cp949')


a = df['RANK']
b = df['타율']
c = df['OPS']
d = df['0.1WAR']
e = df['0.003WRC+']

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


plt.rcParams['font.family'] = 'NanumGothicOTF'

dff = pd.DataFrame({'RANK':a_list, '타율':b_list, 'OPS':c_list, '0.003WRC+':e_list})
ax = dff.plot(x='RANK', y=['타율', 'OPS', '0.003WRC+'], marker='o')
plt.xticks(np.arange(0,20,1))
ax.set_xlabel('RANK')
ax.set_ylabel('Unit')

ax.legend(['타율', 'OPS', '0.003WRC+'], loc='upper left')

ax.set_title('타율과 OPS 및 WRC+의 상관관계')

plt.show()
