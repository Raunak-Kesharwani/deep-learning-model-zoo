import torch 

def conv2d_single_channel(image, kernel, stride:int  = 1 ):
    """
    image  : torch.Tensor of shape (H, W) or (1, H, W)
    kernel : torch.Tensor of shape (kH, kW)
    stride : int
    return : torch.Tensor of shape (H_out, W_out)
    """
# H stands for height and W stands for width 
    # ensure image is (H, W)
    if image.dim == 3 :
        image.squeeze_(0)

    H , W = image.shape
    kH , kW = kernel.shape
    H_out = (H - kH) // stride + 1 
    W_out = (W - kW) // stride + 1 
    output = torch.zeros((H_out,W_out))

    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            w_start = w * stride
            
            patch = image[h_start:h_start+kH, w_start:w_start+kW]
            output[h,w] = torch.sum(patch * kernel)

    return output    


def conv2d_multi_filter(image, kernels, stride=1):
    """
    image   : torch.Tensor of shape (H, W) or (1, H, W)
    kernels : torch.Tensor of shape (F, kH, kW)
    stride  : int

    return  : torch.Tensor of shape (F, H_out, W_out)
    """

    # ensure image is (H, W)
    if image.dim() == 3:
        image = image.squeeze(0)

    H, W = image.shape
    F, kH, kW = kernels.shape

    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    output = torch.zeros((F, H_out, W_out))

    for f in range(F):               # loop over filters
        kernel = kernels[f]

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride

                patch = image[h_start:h_start + kH,
                              w_start:w_start + kW]

                output[f, i, j] = torch.sum(patch * kernel)

    return output
