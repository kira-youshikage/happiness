#主成分降维
"""
第一次修改内容：
    代码规范化
    矩阵乘法规范化
"""
import numpy as np
import pandas as pd

def PCA(X):
	X = np.mat(X)
	#求数据矩阵X的样本协方差矩阵
	s = np.cov(X.T)
	#求样本协方差矩阵的特征值与特征向量
	#a是特征值矩阵，b是经过正交标准化的特征向量矩阵
	a,b = np.linalg.eig(s)
	#调整a使得特征值从小到大排列，同时b也改变
	a_sort,b_sort = sort(a,b)
	#系数矩阵
	A = b_sort.T
	#主成分矩阵
	#Y = A*X.T
	Y = np.dot(A,X.T)
	a_cc = cumulative_contribution(a)
	#依次返回主成分矩阵，系数矩阵，特征值矩阵，累计贡献度矩阵
	return Y.T,A,a_sort,a_cc

#调整特征值矩阵a使得特征值从小到大排列，同时特征向量矩阵b也改变
def sort(a,b):
	t = np.vstack((a,b))
	#矩阵的行数与列数
	size_r,size_c = t.shape
	record = np.zeros((size_r,1))
	#排序
	for i in range(size_c):
		for j in range(i,size_c-1):
			if t[0,i] < t[0,j+1]:
				record = 1*t[:,i]
				t[:,i] = t[:,j+1]
				t[:,j+1] = record
	a_sort = t[0,:]
	b_sort = t[1:,:]
	return a_sort,b_sort

#计算累计贡献度
def cumulative_contribution(a):
	a = np.abs(a)
	a_all = 0
	for i in range(len(a)):
		a_all = a_all + a[i]
	for i in range(len(a)):
		if i == 0:
			a[i] = a[i] / a_all
		else:
			a[i] = a[i - 1] + (a[i] / a_all)
	return a


