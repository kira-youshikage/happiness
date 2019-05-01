import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rad
from sklearn.linear_model import LinearRegression as line
import math
import matplotlib.pyplot as plt
import PCA
from normalization import min_max_scale
import ANOVA
import ann

# 读取
path_train = 'happiness_train_complete.csv'
path_predict = 'happiness_test_complete.csv'
train = pd.read_csv(open(path_train))
predict = pd.read_csv(open(path_predict))

# 清洗
# 删去id,county,survey_time,edu_other
del_list = ['id','county','survey_time','edu_other']
for item in del_list:
	del train[item]
a = []
for item in train.columns:
	if item != 'happiness' and item not in del_list:
		a.append(list(train[item]))
X = []
for i in range(len(train)):
	for j in range(len(a)):
		if type(a[j][i]) == type(' ') or a[j][i] < 0 or math.isnan(a[j][i]):
			X.append(1)
		else:
			X.append(a[j][i])
train_x = np.reshape(X,[int(len(X) / len(a)), len(a)])
y = list(train.happiness)
train_y = np.reshape(y,[len(y),1])
a = []
for item in predict.columns:
	if item != 'happiness' and item not in del_list:
		a.append(list(predict[item]))
X = []
for i in range(len(predict)):
	for j in range(len(a)):
		if type(a[j][i]) == type(' ') or a[j][i] < 0 or math.isnan(a[j][i]):
			X.append(1)
		else:
			X.append(a[j][i])
predict_x = np.reshape(X,[int(len(X) / len(a)),len(a)])

# 把happiness小于1和大于5的数据删掉
happiness_list = [1,2,3,4,5]
train_x_list = []
train_y_list = []
for i in range(train_x.shape[0]):
	if train_y[i,0] in happiness_list:
		train_x_list.append(train_x[i,:])
		train_y_list.append(train_y[i,:])
train_x = np.reshape(train_x_list,
	[len(train_x_list),len(train_x_list[0])])
train_y = np.reshape(train_y_list,
	[len(train_y_list),len(train_y_list[0])])



# 标准化
all_score = np.row_stack((train_x,predict_x))
all_score = min_max_scale(all_score)
train_x = np.reshape(all_score[:train_x.shape[0],:],train_x.shape)
predict_x = np.reshape(all_score[train_x.shape[0]:,:],predict_x.shape)



# 降维
all_PCA,A,a_sort,a_cc = PCA.PCA(
	np.row_stack((np.mat(train_x),predict_x)))
train_PCA_x = all_PCA[:train_x.shape[0],:]
predict_PCA_x = all_PCA[train_x.shape[0]:,:]
n_characters_PCA = 0
for i in range(len(a_cc)):
	if a_cc[i] >= 0.9:
		n_characters_PCA = i+1
		break
train_x = np.mat(train_PCA_x[:,:n_characters_PCA])
predict_x = np.mat(predict_PCA_x[:,:n_characters_PCA])


'''
# 方差分析
train = np.reshape(np.column_stack((train_x,train_y)),
	[train_x.shape[0],train_x.shape[1] + 1 ])
anova_results = ANOVA.ANOVA(train)
train_x_list = []
predict_x_list = []
for i in range(train.shape[1]-1):
	index = 'A' + str(i)
	F = anova_results['F'][index]
	if F > 18:
		train_x_list.append(list(train_x[:,i]))
		predict_x_list.append(list(predict_x[:,i]))
train_x = np.reshape(train_x_list, 
	[len(train_x_list[0]), len(train_x_list)])
predict_x = np.reshape(predict_x_list, 
	[len(predict_x_list[0]), len(predict_x_list)])
'''


# 分割训练集和测试集
model_train_x = np.mat(train_x[:train_x.shape[0] - 1,:])
model_train_y = np.mat(train_y[:train_x.shape[0] - 1,:])
model_test_x = np.mat(train_x[train_x.shape[0] -1:,:])
model_test_y = np.mat(train_y[train_x.shape[0] - 1:,:])



# 线性模型
model = line()
model.fit(model_train_x,model_train_y)
print(model.score(model_test_x,model_test_y))
predict_y = model.predict(predict_x)
index_1 = list()
for i in range(len(predict)):
	index_1.append(i + 8001)
final = pd.DataFrame(data = predict_y, columns = ['happiness'],index = index_1)
path_result = \
	'line.csv'
final.to_csv(path_result,index_label = 'id')



# ann模型
solver = 'sgd'
alpha = 1.7
hidden_layer_sizes = [20,15,10,5]
random_state = 7
clf,_ = ann.regression(model_train_x,model_train_y,
	model_test_x,model_test_y,solver,alpha,hidden_layer_sizes,random_state)
predict_y = clf.predict(predict_x)
index_1 = list()
for i in range(len(predict)):
	index_1.append(i + 8001)
final = pd.DataFrame(data = predict_y, columns = ['happiness'],index = index_1)
path_result = \
	'ann.csv'
final.to_csv(path_result,index_label = 'id')


# 绘图
ann_y = clf.predict(train_x)
line_y = model.predict(train_x)
plt.figure(1)
# 第一张图的第一子图
plt.subplot(311)
plt.plot(model_test_y,'o')
# 第一张图的第二子图
plt.subplot(312)
plt.plot(model.predict(model_test_x),'o')
# 第一张图的第三子图
plt.subplot(313)
plt.plot(clf.predict(model_test_x),'o')
plt.show()


'''
记录：
	特征工程前：line_loss = 0.24576242184289487, ann_loss = 0.5173189826834785
	min-max标准化：line_loss = 0.2458040438085346, ann_loss = 0.26981267847420715
	min-max + LOF：line_loss = 0.2313426729695517, ann_loss = 0.27230121750147773
	min-max + LOF + anova：line_loss = 0.0036758904425988614, ann_loss = 0.3772773163555718
'''