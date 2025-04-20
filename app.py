#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:36:50 2025

@author: chanathiptee
"""

import streamlit as st
import pickle 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

st.set_page_config(page_title="K-Means Clustering", layout="centered")

with open('dtm_trained_model.pkl', 'rb') as f:
    load_model = pickle.load(f)

centers = len(load_model.cluster_centers_)

X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.60, random_state=0)

y_kmeans = load_model.predict(X)

fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(load_model.cluster_centers_[:, 0], load_model.cluster_centers_[:, 1], s=300, c='red')
ax.set_title('k-Means Clustering')
ax.legend()
st.title("K-Means Clustering Visualizer by Chanathip Sirisrisermwong")
st.pyplot(fig)
