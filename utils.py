import torch 

def enrol(x_enrol, y_enrol, x_test, model, device):

    classes = torch.unique(y_enrol)
    n_classes = len(classes)


    def supp_idxs(c):
        '''
        select the n_support first elem in target_cpu that are equal to c
        input: integer 
        output: torch.tensor of size torch.Size([5])
        '''
        # s
        return y_enrol.eq(c).nonzero().squeeze(1)
    
    prototypes = torch.stack([
        
    ])
    for c in classes = 