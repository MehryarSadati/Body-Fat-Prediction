import numpy as np
import sklearn.linear_model as li
import sklearn.neural_network as nn
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.decomposition as dec
import matplotlib.pyplot as plt
import pandas as pd

def result(model, trX, teX, trY, teY):
    r2train = model.score(trX, trY)
    r2test = model.score(teX, teY)
    
    print(f"Train R2 score: {round(r2train, 6)}")
    print(f"Test R2 score: {round(r2test, 6)}")

    trpred = model.predict(trX)
    tepred = model.predict(teX)
    
    a = min([np.min(trpred), np.min(tepred), 0])
    b = max([np.min(trpred), np.min(tepred), 1])

    plt.subplot(1, 2, 1)
    plt.scatter(trY, trpred ,s=12, c='teal')
    plt.plot([a, b], [a, b], c='crimson', lw=1.2, label='y = x')
    plt.title(f'Train [R2 = {round(r2train, 4)}]')
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(teY, tepred ,s=12, c='teal')
    plt.plot([a, b], [a, b], c='crimson', lw=1.2, label='y = x')
    plt.title(f'Test [R2 = {round(r2test, 4)}]')
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.show()
    
df = pd.read_csv('bodyfat.csv', sep=',', header= 0, encoding='utf-8')


data = df.to_numpy()
Y = df['BodyFat'].to_numpy().reshape(-1, 1)
df.drop(['BodyFat'], inplace=True, axis= 1)
X = df.to_numpy()

trX,teX ,trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=0)

scalerX = pp.MinMaxScaler()
trX2 = scalerX.fit_transform(trX)
teX2 = scalerX.transform(teX)

scalerY = pp.MinMaxScaler()
trY2 = scalerY.fit_transform(trY)
teY2 = scalerY.transform(teY)

pca = dec.PCA(n_components=0.95)
trX3 = pca.fit_transform(trX2)
teX3 = pca.transform(teX2)

lr = li.LinearRegression()
lr.fit(trX3, trY2)

mlp = nn.MLPRegressor(hidden_layer_sizes=(30, 40), activation='relu', random_state=0)
mlp.fit(trX3, trY2)

result(lr, trX3, teX3, trY2, teY2)
result(mlp, trX3, teX3, trY2, teY2)