#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:43:55 2022

@author: quindaly
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

###### Create NMF Recommender Class from scratch ######

class NMFRecommender:

    def __init__(self, random_state=15, tol=1e-3, maxiter=200, rank=3):
        """The parameter values for the algorithm"""
        self.random_state = random_state
        self.tol = tol
        self.maxiter = maxiter
        self.rank = rank
  
       
    def initialize_matrices(self, m, n):
        """Initialize the W and H matrices"""
        # Set random seed
        np.random.seed(self.random_state)
        
        # Initialize W and H using random numbers between 0 and 1
        self.W = np.random.random((m, self.rank))
        self.H = np.random.random((self.rank, n))
        
        return self.W, self.H

        
    def compute_loss(self, V, W, H):
        """Computes the loss of the algorithm according to the frobenius norm"""
        loss = np.linalg.norm(V - (W @ H), ord='fro')
        
        return loss

    
    def update_matrices(self, V, W, H):
        """The multiplicative update step to update W and H"""
        # Initialize
        new_H = H
        new_W = W
        
        # Perform the update step once
        # Update H first
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                new_H[i,j] = H[i,j] * ((W.T @ V)[i,j] / (W.T @ W @ H)[i,j])
         # Use the update from H to update W       
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                new_W[i,j] = W[i,j] * ((V @ new_H.T)[i,j] / (W @ new_H @ new_H.T)[i,j])           
        
        return new_W, new_H

      
    def fit(self, V):
        """Fits W and H weight matrices according to the multiplicative update 
        algorithm. Return W and H"""
        
        # Initialize
        m,n = V.shape
        W, H = self.initialize_matrices(m, n)
        diff = np.inf
        iterations = 0
        
        # Run the update until either condition is met
        while diff > self.tol and iterations < self.maxiter:
            new_W, new_H = self.update_matrices(V,W,H)
            
            # increment
            diff = self.compute_loss(V, new_W, new_H)
            H = new_H
            W = new_W
            iterations += 1
            
        return new_W, new_H
        

    def reconstruct(self, W, H):
        """Reconstructs the V matrix for comparison against the original V 
        matrix"""
        
        return W @ H


### Test out the class on the grocery store example ###
        
def groceries():
    """Run NMF recommender on the grocery store example"""
    V = np.array([[0,1,0,1,2,2], # Rows are grocery items
                  [2,3,1,1,2,2], # Columns are customers
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])
    
    # Fit the NMFRecommender to the data V
    nmf = NMFRecommender(rank=2)
    W, H = nmf.fit(V)
    
    # Find number of people with higher weights for comp 2
    higher_2 = 0
    for j in range(H.shape[1]):
        if H[1,j] > H[0,j]:
            higher_2 += 1
    
    return W, H, higher_2


def rank_calculation(file='artist_user.csv'):
    """
    Calculate and return the optimal rank of the specified file
    if the rank exists.
    """
    # Read the data
    df = pd.read_csv(file,index_col=0)
    
    # Calculate benchmark value
    benchmark = np.linalg.norm(df, ord='fro') * 0.0001
    
    # Iterate through various values of rank to find optimal
    rank = 3
    while True:
        
        # initialize the model
        model = NMF(n_components=rank, init='random', random_state=0, max_iter=500)
        W = model.fit_transform(df)
        H = model.components_
        V = W @ H
        
        # Calculate RMSE of original df and new V
        RMSE = np.sqrt(mean_squared_error(df, V))
        
        if RMSE < benchmark:
            return rank, V
        
        # Increment rank if RMSE isn't smaller than the benchmark
        rank += 1
    
### Spotify recommender ###
def discover_weekly(userid):
    """
    Create the recommended weekly 30 new artists list for a given user
    """
    # Load the data
    df = pd.read_csv('artist_user.csv', index_col=0)
    artist_df = pd.read_csv('artists.csv', index_col=0)
    
    # Decompose the dataset using sklearn NMF
    model = NMF(n_components=13)
    model.fit(df)
    
    # Turn H, W, and V into pandas df's for future use
    H = pd.DataFrame(model.components_)    
    W = pd.DataFrame(model.transform(df))    
    V = pd.DataFrame(np.dot(W,H), columns=df.columns)
    V.index = df.index
    
    # Find the row corresponding to the target user, sort it
    user_row = V.loc[userid]    
    user_row = user_row.sort_values(ascending=False)
    
    # Get the top 30 artists that the user hasn't listened to
    top_30 = []    
    for artist in user_row.index:
        if df[artist].loc[userid] == 0: # Listen count is 0
            top_30.append(artist)
            
        if len(top_30) == 30: # Quit when we have the top 30
            break
        
    # Get the names of the artists based on their artist ID
    top_30_artists = []
    for artist in top_30:
        top_30_artists.append([artist_df.loc[int(artist)][0]])
    
    return top_30_artists
    
    
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  