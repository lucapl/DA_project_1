import pandas as pd
import numpy as np

from typing import List

def find_dominated_alternatives(data:pd.DataFrame,isGain:List[bool],domination=lambda a,b:not a>=b):

    a = len(data)
    
    dominates = [[] for _ in range(a)]

    for a_i in range(a):
        for a_j in range(a):
            if a_i == a_j:
                break

            i_dominates_j = True
            for g_j,g in enumerate(data):
                g_vals = np.array(data[g])

                if not isGain[g_j]:
                        
                    g_vals *= -1

                val_i = g_vals[a_i]
                val_j = g_vals[a_j]

                if domination(val_i,val_j):
                    i_dominates_j = False
                    break
        
            if i_dominates_j:
                dominates[a_i].append(a_j)

    return dominates

def show_domination(dominates,names,title="i dominates:",pref=">"):
    print(title)
    for i,dominated in enumerate(dominates):
        print(f"{names[i]} {pref} ",end="")
        print(*map(lambda j: names[j],dominated),sep=", ")

def show_preferences(prefs,names,id_column="id",pref_column = "preferred_to"):
    for i,j in zip(prefs[id_column],prefs[pref_column]):
        print(f"{names[i]} is preferred to {names[j]}")

def verify_preferences(prefs,values,names,id_column="id",pref_column = "preferred_to"):
    for i,j in zip(prefs[id_column],prefs[pref_column]):
        if values[i] > values[j]: # for rankings
            print(f"{names[i]} is not preferred to {names[j]}!")

def rank_string(ranking,names,third_column=None,third_name="id"):
	_str = f'{"rank":<5} | {"name":30} | {third_name:<5}\n'
	_str += len(_str)*"="
	_str += "\n"
	# for i,n in sorted(zip(ranking,names),key=lambda a:a[0]):
	for i,_id in enumerate(ranking):
		n = names[_id]
		_str += f"{i:<5} | {n:<30} | {_id if third_column is None else third_column[_id]:<5}\n"
	return _str

def show_rank(ranking,names):
	print(rank_string(ranking,names))

def flip_rank(rank):
    flipped = [None for id in rank]
    for r,id in enumerate(rank):
        flipped[id] = r
    return flipped