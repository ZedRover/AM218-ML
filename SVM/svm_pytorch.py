#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# In[ ]:





# In[3]:


def train(X, Y, model, args):
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    N = len(Y)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0

        for i in range(0, N, args.batchsize):
            x = X[perm[i : i + args.batchsize]].to(args.device)
            y = Y[perm[i : i + args.batchsize]].to(args.device)

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y * output, min=0))
            loss += args.c * (weight.t() @ weight) / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))


# In[1]:


get_ipython().system('pip install pandoc')


# In[ ]:





# In[ ]:





# In[ ]:




