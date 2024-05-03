import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
import pandas as pd

from queue import Queue

from src.utils import rank_string,show_rank,flip_rank

from typing import Tuple,List,Any

def type_1(d):
	return 1 if d > 0 else 0

def type_5(d,q:float,p:float):
	if d > p:
		return 1
	elif d <= q:
		return 0
	else: 
		return (d-q)/(p-q)

class Promethee:
	def __init__(self,data:pd.DataFrame,criterion_type:Tuple[bool],criterion_weights:Tuple[float]=None,discrimination_thresholds:np.array=None):
		self.data = data
		self.names = data.index
		self.ranking = [i for i in range(len(self.names))]
		self.is_crit_gain = criterion_type
		self.criterion_weighs = criterion_weights
		self.criterion_funcs = self._init_crit_funcs(discrimination_thresholds)

	def _init_crit_funcs(self,disc_thresh):
		output=[]
		for i,(p,q) in enumerate(disc_thresh):
			output.append((lambda qi,pi: lambda a,b: type_5(a-b,qi,pi))(q,p) if self.is_crit_gain[i] else (lambda qi,pi: lambda a,b: type_5(b-a,qi,pi))(q,p))
		return output
		

	def _cp_index(self,data:pd.DataFrame,criterion_weights:List[float]=None,criterion_func:List[callable]=None)->np.array:

		n = len(data)
		matrix = np.zeros((n,n))

		if criterion_weights==None:
			criterion_weights = [1 for _ in data.columns]

		if criterion_func == None:
			criterion_func = [type_1 for _ in criterion_weights]

		for c,criterion in enumerate(data.columns):
			for i,a in enumerate(data[criterion]):
				for j,b in enumerate(data[criterion][i+1:],i+1):
					pi = criterion_func[c]
					w = criterion_weights[c]
					matrix[i][j] += w * pi(a,b)
					matrix[j][i] += w * pi(b,a)
		
		return matrix/sum(criterion_weights)
	
	def _flow(self,cpi:np.array):
		positive_flow = np.sum(cpi,axis=1)
		negative_flow = np.sum(cpi,axis=0)

		return positive_flow,negative_flow
	
	def _flip_rank(self,rank):
		return flip_rank(rank)

	def flipped_ranking(self):
		return self._flip_rank(self.ranking)

	def __str__(self):
		return rank_string(self.ranking,self.names)
	
class Promethee1(Promethee):
	def rank(self):
		cpi = self._cp_index(self.data,self.criterion_weighs,self.criterion_funcs)
		self.pos_flow,self.neg_flow = self._flow(cpi)

		self.pos_rank = np.argsort(-self.pos_flow)
		self.neg_rank = np.argsort(self.neg_flow)

		pos_pref_matrix = self._pref_from_ranking(self.pos_rank)
		neg_pref_matrix = self._pref_from_ranking(self.neg_rank)

		combined_matrix = np.logical_and(pos_pref_matrix,neg_pref_matrix).astype(np.uint8)

		self._simplify_ranking(combined_matrix)
		
		self._show_ranking(combined_matrix)

		return self.ranking
	
	def _show_ranking(self,matrix):
		g = nx.DiGraph(matrix)
		pos = nx.drawing.nx_agraph.graphviz_layout(g, prog='dot')
		plt.figure(figsize=(10,10))
		nx.draw(g,pos,with_labels=False)
		custom_labels = self.names
		nx.draw_networkx_labels(g, pos, labels=custom_labels, font_size=12, font_color='black')

	def _simplify_ranking(self,matrix):
		#for i, row in enumerate(matrix):
		for i in range(len(matrix)):
			matrix = self._remove_transient_connections(matrix,i)
			
			

	def _remove_transient_connections(self,matrix,start):
		#bfs 
		q = Queue()
		q.put(start)
		isQueued = [False for _ in matrix]
		while not q.empty():
			i = q.get()
			#neighbours = np.where(matrix[i]==1)
			row = matrix[i]
			for j,n in enumerate(row):
				if n == 0: continue

				if i != start and matrix[start,j] == 1:
					matrix[start,j] = 0

				#if isQueued[j]
				q.put(j)

		return matrix
	

	def _pref_from_ranking(self,ranking):
		a = len(self.data)
		pref_matrix = np.zeros((a,a))	

		for r_i,i in enumerate(ranking):
			for r_j,j in enumerate(ranking[r_i+1:],r_i):
				pref_matrix[i][j] = 1

		return pref_matrix


	def __str__(self):
		return f"Positive flow ranking:\n{self.positive_ranking()}\nNegative flow ranking:\n{self.negative_ranking()}"
	
	def positive_ranking(self):
		return rank_string(self.pos_rank,self.names)
	def negative_ranking(self):
		return rank_string(self.neg_rank,self.names)

	def _pref_check(self,i,j):
		return self.pos_flow[i] >= self.pos_flow[j] and self.neg_flow[i] <= self.neg_flow[j]
	def _equal_check(self,i,j):
		return (np.allclose(self.pos_flow[i],self.pos_flow[j]) and np.allclose(self.neg_flow[i],self.neg_flow[j]))
	def _incomparable(self,i,j):
		return ((self.pos_flow[i] > self.pos_flow[j] and self.neg_flow[i] > self.neg_flow[j]) or 
				(self.pos_flow[i] < self.pos_flow[j] and self.neg_flow[i] < self.neg_flow[j]))
	
class Promethee2(Promethee):
	def rank(self):
		self.cpi = self._cp_index(self.data,self.criterion_weighs,self.criterion_funcs)
		pos_flow,neg_flow = self._flow(self.cpi)

		self.f = pos_flow-neg_flow

		self.ranking = np.argsort(-self.f)

		return self.ranking