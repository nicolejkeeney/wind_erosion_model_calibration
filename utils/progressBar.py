import sys

def progressBar(i, tot): 
    """Display a progress bar inside a for loop 
    
    Args: 
        i (int): iteration number
        tot (int): total number of iterations
    """
    
    j = (i + 1) / tot
    sys.stdout.write('\r [%-20s] %d%% complete' % ('='*int(20*j), 100*j))
    sys.stdout.flush()  
