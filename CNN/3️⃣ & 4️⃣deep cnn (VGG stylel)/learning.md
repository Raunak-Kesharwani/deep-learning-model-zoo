# Learning from this project 

Trained Deep CNN without normlization with adam in VGG style  
![image.png](attachment:103dcabb-f497-4a2e-b803-5edb12b27c84.png)

we can not pass value directly in nn.class because nn contains whatever they are class / constructor  

use nn.dropout() instead of nn.dropout1d() this is for CNN1d 
use dropout after activation (standard good practice)
weight decay standard for Sgd 5e-4

# Observations 

Observed when training started test accuracy is started from 45% when using SGD :  
    Accuracy: 79.6%  
    Avg loss: 101.463741  
    # this is the clearly case of overfitting training accuracy is too high(95%) training_loss is low but better than adam with no normalization  
    # but test_accuracy is low and test_loss is high   

obsereved training unstabilty may be because of adam optimizer :
    Accuracy: 78.0%  
    Avg loss: 164.200599  
    # this is the clearly case of overfitting training accuracy is too high(95%) training_loss is low with no normalization
    # but test_accuracy is low and test_loss is high 

# After normalization 
with SGD got :
    best validation Accuracy: 84.3%  
    lowest Avg loss: 101.387872  
    but training accuracy is still high means we can still impove model  
    # after adding L2 regularization 

with Adam got :
    best validation Accuracy: 83% 
    but training accuracy is still high means we can still impove model  
    got with lr 2e-3
     Accuracy: 84.1%
    Avg loss: 104.685413
    # after adding L2 regularization
    Accuracy: 84.7%
    Avg loss: 85.936681

# After adding correct dropout nn.Dropout()

got to see imrovement 
with adam got :
    best validation accuracy : 85.7%

after applying on the fly augumenttion got to see imporve by 1% and overfitting is completely reduced 
    Accuracy: 86.0%
    Avg loss: 64.976720

after incresing epoches to 80 :
    Accuracy: 91.5%
    Avg loss: 51.88400
 