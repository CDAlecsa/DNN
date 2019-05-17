'''
            Plot functions for the accuracy and for the loss function
'''


from numpy import arange
import matplotlib.pyplot as plt



#%%%%%%%%%%%%%%%%%%%%%%%    Accuracy plot   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
   
def plot_accuracy(number_of_epochs, train_acc, val_acc=[]):
    plt.figure()
    
    plt.plot(arange(1, number_of_epochs+1), train_acc[0:number_of_epochs], 'ro-', label='train_acc')
    if val_acc != []:
        print(len(arange(1, number_of_epochs+1)),len(val_acc[0:number_of_epochs]))
        plt.plot(arange(1, number_of_epochs+1), val_acc[0:number_of_epochs], 'bs-', label='val_acc')
    
    plt.xlim([0, number_of_epochs])       
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
  


#%%%%%%%%%%%%%%%%%%%%%%%    Loss plot   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

def plot_loss(number_of_epochs, train_loss, val_loss=[]):
    plt.figure()
    
    plt.plot(arange(1, number_of_epochs+1), train_loss[0:number_of_epochs], 'ro-', label='train_loss')
    if val_loss != []:
        plt.plot(arange(1, number_of_epochs+1), val_loss[0:number_of_epochs], 'bs-', label='val_loss')
    
    plt.xlim([0, number_of_epochs])       
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()  