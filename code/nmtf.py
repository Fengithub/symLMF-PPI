import numpy as np
from numpy.linalg import norm, multi_dot
import scipy as sp
import scipy.sparse
import sys
import pdb
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from scipy.sparse import csr_matrix
from sklearn import metrics
import gc
import random

class NMTF:

	def __init__(self, k=30, max_iter=75, max_step=500):
		self.k = k
		self.max_iter = max_iter
		self.max_step = max_step

	def train_model(self, Z, train_data, m, seed):
		random.seed(seed)

		#random samples from a uniform distribution over [0,1)
		self.H = 0.5 * np.random.rand(m,self.k)
		self.S = 0.5 * np.random.rand(self.k,self.k)

		plusval = 1e-8  # make zero plus something for divide
		iteration = 0
		while iteration < self.max_iter:
			step = 0
			while step < self.max_step:
				# update H
				H_top = multi_dot([Z,self.H,self.S])
				H_bottom = multi_dot([self.H,self.S,self.H.T,self.H,self.S]) + plusval
				# update S
				S_top = multi_dot([self.H.T, Z, self.H])
				S_bottom = multi_dot([self.H.T,self.H,self.S,self.H.T,self.H]) + plusval

				self.H *= (H_top / H_bottom) ** 0.25
				self.S *= S_top / S_bottom
				step += 1

			# update Z
			Z = multi_dot([self.H, self.S, self.H.T])
			for i, j, v in train_data:
				Z[i][j] = v
				Z[j][i] = v

			iteration += 1

		self.Y = multi_dot([self.H, self.S, self.H.T])
		self.Y = np.clip(self.Y, 0, 1)
		self.Y = csr_matrix(self.Y)

	def evaluation(self, test_data):
		scores = []
		for i, j, _ in test_data:
			scores.append(self.Y[i, j])
		auc_val = roc_auc_score(test_data[:, 2], scores)
		prec, rec, _ = precision_recall_curve(test_data[:, 2], scores)
		aupr_val = auc(rec, prec)
		return aupr_val, auc_val

	def predict_scores(self, test_data):
		scores = []
		for i, j, _ in test_data:
			scores.append(self.Y[i, j])
		return np.round(scores, 4)