import numpy as np

def norm_by_param(vec, model):
    results = {}
    i = 0
    for name, p in model.named_parameters():
        n = p.numel()
        if p.requires_grad:
            results[name] = np.linalg.norm(vec[i:i+n])/n
        
            i += n
        else:
            results[name] = 0
        
#     for j, p in enumerate(model.parameters()):
#         n = p.numel()
#         if p.requires_grad:
#             results[str(j)] = np.linalg.norm(vec[i:i+n])
#             i += n
#         else:
#             results[str(j)] = 0
    
        
    
    return results

# def norm_by_param(vec, model):
#     results = {}
#     i = 0
#     for name, p in model.named_parameters():
#         n = p.numel()
#         results[name] = np.linalg.norm(vec[i:i+n])
    
#         i += n
    
#     return results