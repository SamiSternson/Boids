"""This file is just to test out functions that don't rely on the main boid simulation, so theres no real reason too look at it."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numba import njit
matplotlib.use('TkAgg')
@njit
def chose_index_based_on_inverse_power(k, p):
    probs=[(1/(i+1))**p for i in range(k)]
    new_probs=[probs[i]*(100/sum(probs)) for i in range(k)]
    print(new_probs)
    for _ in range(k):
        rand=np.random.random()*100
        if rand>(100-sum(new_probs[0:_+1])):
            return _
    return 0

num_visible_boids=8
n=10000
neighbors=[0 for i in range(num_visible_boids)]
for i in range(n):
    j=chose_index_based_on_inverse_power(num_visible_boids, 2)
    neighbors[j]+=1
plt.plot([i for i in range(num_visible_boids)], neighbors)
plt.xlabel("Index of sorted list of neighbor boids")
plt.ylabel("% of times this index was chosen")
plt.show()