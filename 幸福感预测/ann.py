# ann
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# 计算代价
def get_loss(predict_y,target_y):
	loss = 0
	# 训练样本数
	number = predict_y.shape[0]
	for i in range(number):
		loss += (predict_y[i,0] - target_y[i,0])**2
	loss /= 2* number
	return loss
# 计算正确率（离散）
def accuracy(predict_y,target_y):
	n_true = 0
	n = predict_y.shape[0]
	for i in range(n):
		if predict_y[i,0] == target_y[i,0]:
			n_true += 1
	result = n_true / n
	return result

# MLP模型(分类)
def classifier(train_x,train_y,test_x,test_y,solver,alpha,hidden_layer_sizes,random_state):
	'''
	solver:MLP的求解方法("lbfgs","adam","sgd")
		L-GBFS在小数据上表现较好，adam较为鲁棒，
		SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）：SGD标识随机梯度下降。
	alpha:L2的参数：MPL是可以支持正则化的，默认为L2，具体参数需要调整
	hidden_layer_sizes=(5,2)：hidden层2层，第一层5个神经元，第二层2个神经元
	'''
	clf = MLPClassifier(solver = solver,alpha = alpha,
		hidden_layer_sizes = hidden_layer_sizes,random_state = random_state)
	clf.fit(train_x,train_y)
	predict_y = np.mat(clf.predict(test_x)).T
	result = accuracy(predict_y,test_y)
	print("accuracy = ",end = '\t')
	print(result)
	return clf,result

def regression(train_x,train_y,test_x,test_y,solver,alpha,hidden_layer_sizes,random_state):
	'''
	solver:MLP的求解方法("lbfgs","adam","sgd")
		L-GBFS在小数据上表现较好，adam较为鲁棒，
		SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）：SGD标识随机梯度下降。
	alpha:L2的参数：MPL是可以支持正则化的，默认为L2，具体参数需要调整
	hidden_layer_sizes=(5,2)：hidden层2层，第一层5个神经元，第二层2个神经元
	'''
	clf = MLPRegressor(solver = solver,alpha = alpha,
		hidden_layer_sizes = hidden_layer_sizes,random_state = random_state)
	clf.fit(train_x,train_y)
	predict_y = np.mat(clf.predict(test_x)).T
	loss = get_loss(predict_y,test_y)
	print("loss = ",end = '\t')
	print(loss)
	return clf,loss


