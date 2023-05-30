# -*- coding: utf-8 -*-
"""
@author: david
"""

import torch.nn as nn

class SmallNetwork(nn.Module):
    def __init__(self, inputSize, numClasses, fc1,fc2, dropout):
        super().__init__()
        
        # Linear transformations - hidden Layers and Output Layer
        self.hiddenLay1 = nn.Linear(inputSize,fc1)
        self.hiddenLay2 = nn.Linear(fc1,fc2)

        self.output = nn.Linear(fc2, numClasses)
        
        # Define tanH activation and softmax for output 
        self.tanHAct = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        #  Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)
        
        self.batchnorm1 = nn.BatchNorm1d(fc1)
        self.batchnorm2 = nn.BatchNorm1d(fc2)

        # Apply weight initialization
        self.apply(self._init_weights)
        
    def forward(self, x):
        # First Layer - Input with tanH Activation and dropout
        x = self.hiddenLay1(x)
        x = self.batchnorm1(x)
        x = self.tanHAct(x)
        x = self.dropout(x)
        # Second Layer - Hidden Layer with tanH Activation
        x = self.hiddenLay2(x)
        x = self.batchnorm2(x)
        x = self.tanHAct(x)
        x = self.dropout(x)

        # Output Layer
        x = self.output(x)
        x = self.softmax(x)
        
        return x
    
    def _init_weights(self, module):
        # Apply weight initialization
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()