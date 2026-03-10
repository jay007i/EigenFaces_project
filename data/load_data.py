from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from matplotlib import pyplot as plt
def load_faces():
    data = fetch_olivetti_faces()
    x = data.data
    y = data.target
    return x,y


