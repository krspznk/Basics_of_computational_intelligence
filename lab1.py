import numpy as np
import pandas as pd

X=np.array([[0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1]])
T=np.array([0,1,1,1,1,1,1,1])
eta=0.07
theta=0.6
np.random.seed(0)
w=np.random.random(3)
gammas=np.array([0,0,0, 0])
# w=[0.608814,  0.715189,  0.602763]
print(f"Eta = {eta}")
print(f"Theta = {theta}")
data=pd.DataFrame(columns=["w1", "w2", "w3", "theta", "x1", "x2", "x3", "a", "y", "t", "gamma",
                           "gamma_w1", "gamma_w2", "gamma_w3", "gamma_thata"])
data["x1"]=[x[0] for x in X]
data["x2"]=[x[1] for x in X]
data["x3"]=[x[2] for x in X]
data["t"]=T
for i in range(len(X)):
    data.loc[i, "w1"] = w[0]
    data.loc[i, "w2"] = w[1]
    data.loc[i, "w3"] = w[2]
    data.loc[i, "theta"]=theta
    a=sum(X[i]*w)
    data.loc[i,"a"]=a
    if a>theta:
        y=1
    else:
        y=0
    data.loc[i, "y"]=y
    if y!=T[i]:
        gamma=eta*(T[i]-y)
        gammas[0]=X[i][0]*gamma
        gammas[1] = X[i][1] * gamma
        gammas[2] = X[i][2] * gamma
        gammas[3] = theta * gamma
        data.loc[i, "gamma_w1"]=gammas[0]
        data.loc[i, "gamma_w2"] = gammas[1]
        data.loc[i, "gamma_w3"] = gammas[2]
        data.loc[i, "gamma_thata"] = gammas[3]
        w[0] += gammas[0]
        w[1] += gammas[1]
        w[2] += gammas[2]
        theta += gammas[3]

data['gamma'] = data['gamma'].fillna(0)
data['gamma_w1'] = data['gamma_w1'].fillna(0)
data['gamma_w2'] = data['gamma_w2'].fillna(0)
data['gamma_w3'] = data['gamma_w3'].fillna(0)
data['gamma_thata'] = data['gamma_thata'].fillna(0)

print(data.to_string())