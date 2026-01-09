import torch 


def relu_forward(x):
    """
    x : torch.Tensor (any shape)
    return:
        out : ReLU activated tensor
        cache : x (stored for backward)
    """
    out = torch.clamp(x, min=0)
    cache = x
    return out, cache


def maxpool2d_forward(x, pool_size=2, stride=2):
    C, H, W = x.shape
    H_out = H // pool_size
    W_out = W // pool_size

    out = torch.zeros((C, H_out, W_out))
    mask = {}

    for c in range(C):
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                window = x[c, i:i+pool_size, j:j+pool_size]
                max_val = window.max()
                out[c, i//stride, j//stride] = max_val

                # store index
                idx = torch.argmax(window)
                di, dj = idx // pool_size, idx % pool_size
                mask[(c, i//stride, j//stride)] = (i+di, j+dj)

    return out, mask