"""Bound displayed correlation degree while retaining the strongest candidates."""
import math


def limit_correlation_edges(edges,n_nodes,max_neighbors=8):
    if not isinstance(max_neighbors,int) or max_neighbors<1:
        raise ValueError('max_neighbors must be a positive integer')
    candidates={}
    for a,b,weight in edges:
        a,b=int(a),int(b);weight=float(weight)
        if not (0<=a<n_nodes and 0<=b<n_nodes):raise ValueError('Unknown edge endpoint')
        if not math.isfinite(weight):raise ValueError('Correlation weights must be finite')
        if a==b or weight==0:continue
        key=(min(a,b),max(a,b))
        if abs(weight)>abs(candidates.get(key,0)):candidates[key]=weight
    degree=[0]*n_nodes;chosen=[]
    for (a,b),weight in sorted(candidates.items(),key=lambda item:(-abs(item[1]),*item[0])):
        if degree[a]<max_neighbors and degree[b]<max_neighbors:
            chosen.append([a,b,weight]);degree[a]+=1;degree[b]+=1
    return sorted(chosen,key=lambda edge:(edge[0],edge[1]))


def correlation_edge_style(weights,zoom=False):
    """Thin, faint lines with width and opacity proportional to |correlation|."""
    import numpy as np
    strength=np.clip(np.abs(np.asarray(weights,dtype=float)),0.,1.)
    return strength*(.55 if zoom else .30),strength*(.30 if zoom else .14)
