import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np
from pandas.plotting import radviz


chameleon = np.load("./att/att_chameleon.npy")
chameleon=np.mean(chameleon, axis=0)
print(chameleon)
squirrel = np.load("./att/att_squirrel.npy")
squirrel=np.mean(squirrel, axis=0)
print(squirrel)
actor= np.load("./att/att_film.npy")
actor=np.mean(actor, axis=0)
print(actor)


dic = {
    "Chameleon": chameleon,  # 列表
    "Squirrel": squirrel,  # 数组
    "Actor": actor,
}  # 元组
data = pd.DataFrame(dic, index=["0", "1", "2"])  # 创建Dataframe
print(data)
sns.heatmap(data=data,square=True,annot=True)
plt.xlabel("Dataset")
plt.ylabel("Order")
plt.savefig('./att/Webpage network.jpg',dpi=600)
plt.show()