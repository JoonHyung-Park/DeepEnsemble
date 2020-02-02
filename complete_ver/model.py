import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,num_input=1,num_output=1,hidden_layers=[100]):
        super().__init__()
        self.inputs = num_input
        self.outputs = num_output
        self.n_layers = len(hidden_layers)
        self.net_structure = [self.inputs,*hidden_layers,self.outputs]


        for i in range(self.n_layers+1):
            setattr(self,'layer_'+str(i), nn.Linear(self.net_structure[i], self.net_structure[i+1]))

    def forward(self,x):
        for i in range(self.n_layers):
            layer = getattr(self, 'layer_'+str(i))
            x = torch.relu(layer(x))
        layer = getattr(self,'layer_'+str(self.n_layers))
        x = layer(x)
        return x

def NLLloss(y,mean, var):
    return (torch.log(var)+ (y - mean)**2/(2*var)).sum()

class GaussianMLP(MLP):
    def __init__(self,num_input=1, num_output=2, hidden_layers=[100]):
        super().__init__(num_input = num_input,num_output = num_output,hidden_layers= hidden_layers)
        self.inputs = num_input
        self.outputs = num_output
    def forward(self,x):
        for i in range(self.n_layers):
            layer = getattr(self, 'layer_'+str(i))
            x = torch.relu(layer(x))
        layer = getattr(self, 'layer_' + str(self.n_layers))
        x = layer(x)
        mean, variance = torch.split(x, self.outputs-1, dim=1)
        variance = F.softplus(variance) + 1e-6 #Positive constraint
        return mean, variance

class GaussianMixtureMLP(nn.Module):

    def __init__(self, n_models=5, inputs=1,outputs=2, hidden_layers = [100]):
        super().__init__()
        self.n_models = n_models
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        for i in range(self.n_models):
            model = GaussianMLP(self.inputs,self.outputs,self.hidden_layers)
            setattr(self,'model_'+str(i),model)

    def forward(self,x):
        means = []
        variances = []
        for i in range(self.n_models):
            model = getattr(self,"model_"+str(i))
            mean, var = model(x)
            means.append(mean)
            variances.append(var)
        means = torch.stack(means)
        mean = means.mean(dim=0)
        variances = torch.stack(variances)
        var = (variances +  means.pow(2)).mean(0) - mean.pow(2)
        return mean, var

def train_model_step(model, optimizer, x, y):
    """ Training an individual gaussian MLP of the deep ensemble. """
    optimizer.zero_grad()
    mean, var = model(x)
    loss = NLLloss(y, mean, var)
    loss.backward(retain_graph = True)
    data_grad = x.grad.data
    sign_grad = data_grad.sign()
    perturbed_data = x + 0.01*sign_grad
    mean_new, var_new = model(perturbed_data)
    gaussian_loss_new = NLLloss(y, mean_new, var_new)
    loss += gaussian_loss_new
    model.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
