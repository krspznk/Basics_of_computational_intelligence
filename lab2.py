import numpy as np

np.random.seed(133)


class Neuron:
    def __init__(self, layers: list, learning_rate=0.1, iter=1000):
        self.layers = layers
        self.learning_rate = learning_rate
        # значення в нейронах перед активацією
        self.A = [np.random.rand(l + 1, 1) for l in self.layers]
        # значення після активації
        self.Z = [np.random.rand(l + 1, 1) for l in self.layers]
        # вагові коефіціенти
        self.W = [np.random.rand(l, lp + 1) for (lp, l) in zip(self.layers, self.layers[1:])]
        # дельти
        self.D = [np.empty((l, 1)) for l in layers[1:]]

    def train(self, X, Y, iter=1000):
        for i in range(iter):
            error = 0
            for x, t in zip(X, Y):
                self.forward(x)
                self.backward(t)
                # shape(-1) - перетворення в одновимірний масив
                error += self.loss(t, self.output().reshape(-1))
            if not i % 100:
                print('iteration {}: error = {:.5f}'.format(i, error / len(X)))

    def predict(self, X):
        return np.vstack(tuple(np.copy(self.forward(x)) for x in X))

    def forward(self, x: np.array):
        self.A[0][1:] = x.reshape((-1, 1))
        for l in range(2):
            # множення матриць
            z = self.W[l] @ self.A[l]
            self.Z[l + 1][1:] = z
            self.A[l + 1][1:] = self.sigmoid(z)
        return self.output().reshape(-1)

    def backward(self, t):
        t = t.reshape((-1, 1))
        self.D[-1] = self.diff(t, self.output()) * self.sigmoid_back(self.Z[-1][1:])
        for l in range(-2, -len(self.layers), -1):
            self.D[l] = (self.sigmoid_back(self.Z[l]) * self.W[l + 1].T @ self.D[l + 1])[1:]
        for a, w, d in zip(self.A, self.W, self.D):
            w -= self.learning_rate * a.T * d

    def output(self):
        return self.A[-1][1:]

    def loss(self, t, y):
        return np.sum(np.abs(t - y))

    def diff(self, t, y):
        return y - t

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_back(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))



def f(X):
    x1 = X[:,0:1]
    x2 = X[:,1:2]
    x3 = X[:,2:3]

    y1 = (2*(x1**2) + np.cos(x2) - np.sin(x3))
    y2 = np.where(y1 > y1.mean(), 1, 0)

    #з'єднання масивів по горизонталі
    return np.hstack((y1, y2))


layers = [3, 3, 2]
neuron = Neuron(layers)
#20 рядків, 3 стовпці, від 0 до 1
X = np.random.rand(20, 3)
Y = f(X)
Ymax=0;
Ymin=1;
for i in range(20):
    if (Y[i][0]>Ymax): Ymax = Y[i][0]
    if (Y[i][0]<Ymin): Ymin = Y[i][0]
for i in range(20):
    Y[i][0] = (Y[i][0] - Ymin)/(Ymax-Ymin)

neuron.train(X, Y, iter=1000)
predictions = neuron.predict(X)

print('\nAverage error = {:.5f}' .format(np.abs(predictions - Y).mean()))


print('{:<25}{:<25}'.format('\n target', ' predicted'))
for y, t in zip(Y, predictions):
    print('{:<25}{:<25}'.format(str(y), str(t)))

print()
print()

np.random.seed(131)
#5 рядків, 3 стовпці, від 0 до 1
X = np.random.rand(5, 3)
Y = f(X)

for i in range(5):
    if (Y[i][0]>Ymax): Ymax = Y[i][0]
    if (Y[i][0]<Ymin): Ymin = Y[i][0]
for i in range(5):
    Y[i][0] = (Y[i][0] - Ymin)/(Ymax-Ymin)

neuron.train(X, Y, iter=1000)
predictions = neuron.predict(X)

# {:.5f} - число з плаваючою точкою, 5 знаків після коми
print('\nAverage error = {:.5f}' .format(np.abs(predictions - Y).mean()))

# < - по лівому краю
print('{:<25}{:<25}'.format('\n target', ' predicted'))
for y, t in zip(Y, predictions):
    print('{:<25}{:<25}'.format(str(y), str(t)))

