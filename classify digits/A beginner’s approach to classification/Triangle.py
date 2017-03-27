# %matplotlib inline
#要在ipy notebook里直接显示Matplotlib的输出图象，需要%inline

import matplotlib.pyplot as plt
import numpy as np
X = np.linspace(-np.pi,np.pi,256,endpoint=True)
(C,S)=np.cos(X),np.sin(X)

#这里用到了Matplotlib和numpy模块,linspace在(−π,π)之间分成共256个小段，
#并把这256个值赋予X。C,S分别是cosine和sine值（X,C,S都是numpy数组）
plt.plot(X,C)
plt.plot(X,S)

plt.show()