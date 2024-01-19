import torch

#  Adaptive Sum of Squared Errors loss function
def ada_SSE_loss(target, input):
    """Adaptive Sum of Squared Errors Loss Function"""
    loss = ((input-target)**2).sum()
    return(loss)
