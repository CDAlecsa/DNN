'''
                                 Objective functions   
'''

from numpy import sum as SUM
from numpy import where, nan_to_num, log, sqrt
from numpy.linalg import norm





#%%%%%%%%%%%%%%%%%%%%%%%    Quadratic function    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def QUADRATIC_loss(a, y):  
    return 0.5 * norm(a-y) ** 2


def QUADRATIC_loss_derivative(a, y):
    return a - y


def QUADRATIC_loss_error(z, a, y, activ_deriv) :
    return QUADRATIC_loss_derivative(a, y) * activ_deriv(z)
        
    
    
#%%%%%%%%%%%%%%%%%%%%%%%   Cross Entropy function    %%%%%%%%%%%%%%%%%%%%%%%%%%

    
def CROSS_ENTROPY_loss(a, y):
    return SUM(nan_to_num( -y * log(a) - (1-y) * log(1-a) ))



def CROSS_ENTROPY_loss_derivative(a, y):
    return where(a*(1-a)!=0, (a-y) / ( a * ( 1 - a ) ), 0)



def CROSS_ENTROPY_loss_error(z, a, y, activ_deriv):
    return CROSS_ENTROPY_loss_derivative(a, y) * activ_deriv(z)



#%%%%%%%%%%%%%%%%%%%%%%%   Kullback divergence function    %%%%%%%%%%%%%%%%%%%%


def KULLBACK_loss(a, y):
    return SUM( where(a!=0, y * nan_to_num(log(y/a) ) , 0)  )



def KULLBACK_loss_derivative(a, y):
    return where(a!=0, -y/a, 0)



def KULLBACK_loss_error(z, a, y, activ_deriv):
    return KULLBACK_loss_derivative(a, y) * activ_deriv(z)




#%%%%%    Generalized Kullback-Leibler divergence function    %%%%%%%%%%%%%%%%%
    

def KULLBACKLEIBLER_loss(a, y):
    return SUM( where(a!=0, y * nan_to_num(log(y/a) ) , 0)  ) - SUM(y) + SUM(a)



def KULLBACKLEIBLER_loss_derivative(a, y):
    return where(a!=0, (a-y)/a, 0)



def KULLBACKLEIBLER_loss_error(z, a, y, activ_deriv):
    return KULLBACKLEIBLER_loss_derivative(a, y) * activ_deriv(z)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Hellinger function    %%%%%%%%%%%%%%%%%%%%%%%
    

def HELLINGER_loss(a, y):
    return (1/sqrt(2)) * SUM(( sqrt(a) - sqrt(y) ) ** 2)

def HELLINGER_loss_derivative(a, y):
    return (1/sqrt(2)) * where(a!=0, ( sqrt(a) - sqrt(y) ) / sqrt(a), 0)

def HELLINGER_loss_error(z, a, y, activ_deriv):
    return HELLINGER_loss_derivative(a, y) * activ_deriv(z)



#%%%%%%%%%%%%%%%%%%%%%%%  The functions are grouped in some dictionaries %%%%%%
    

loss = {
             'QUADRATIC_loss' : QUADRATIC_loss, 
             'CROSS_ENTROPY_loss' : CROSS_ENTROPY_loss,
             'KULLBACK_loss' : KULLBACK_loss,
             'HELLINGER_loss' : HELLINGER_loss,
             'KULLBACKLEIBLER_loss' : KULLBACKLEIBLER_loss
            }




loss_error = {
             'QUADRATIC_loss_error' : QUADRATIC_loss_error, 
             'CROSS_ENTROPY_loss_error' : CROSS_ENTROPY_loss_error,
             'KULLBACK_loss_error' : KULLBACK_loss_error,
             'HELLINGER_loss_error' : HELLINGER_loss_error,
             'KULLBACKLEIBLER_loss_error' : KULLBACKLEIBLER_loss_error
            }
        