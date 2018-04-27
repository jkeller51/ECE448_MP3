# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 17:41:13 2018

@author: jkell
"""
import DNN
import numpy as np

def load_affine(fname):
    f = open(fname, 'r')
    file = []
    for line in f:
        file.append(line)
    A = []
    W = []
    b = []
    Z = []
    dZ = []
    for i in range(1,11):
        linedata = file[i].split(" ")
        row = []
        for q in linedata:
            try:
                row.append(float(q))
            except: 
                pass
        A.append(row)
        
    A = np.asarray(A)
    
    for i in range(13,21):
        linedata = file[i].split(" ")
        row = []
        for q in linedata:
            try:
                row.append(float(q))
            except: 
                pass
        W.append(row)
        
    W = np.asarray(W)
    
    
    linedata = file[23].split(" ")
    row = []
    for q in linedata:
        try:
            row.append(float(q))
        except: 
            pass
    b = row
        
    b = np.asarray(b)
    
    W = np.vstack([W, b])
    
    
    for i in range(26,36):
        linedata = file[i].split(" ")
        row = []
        for q in linedata:
            try:
                row.append(float(q))
            except: 
                pass
        Z.append(row)
        
    Z = np.asarray(Z)
    
    for i in range(38,48):
        linedata = file[i].split(" ")
        row = []
        for q in linedata:
            try:
                row.append(float(q))
            except: 
                pass
        dZ.append(row)
        
    dZ = np.asarray(dZ)
    
    return A,W,b,Z,dZ


def load_entropy(fname):
    f = open(fname, 'r')
    file = []
    for line in f:
        file.append(line)
    F = []
    y = []
    for i in range(1,11):
        linedata = file[i].split(" ")
        row = []
        for q in linedata:
            try:
                row.append(float(q))
            except: 
                pass
        F.append(row)
        
    F = np.asarray(F)
    
    linedata = file[13].split(" ")
    row = []
    for q in linedata:
        try:
            row.append(float(q))
        except: 
            pass
    y = np.asarray(row)
    
    return F,y


A,W,b,Z,dZ = load_affine('./data/check_affine.txt')


# calculate dL/dW
dW = np.matmul(np.transpose(A), dZ)

dW = np.vstack([dW, np.sum(dZ,axis=0)])   # add bias db

W = W[0:-1,:]
    
da = np.matmul(dZ,np.transpose(W))


F,y = load_entropy('./data/check_cross_entropy.txt')

L, dF = DNN.cross_entropy(DNN.softmax(F),y)