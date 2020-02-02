from utils import *
import argparse
import numpy as np
from model import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'





def main(args):
    x = np.linspace(-6,6,100).reshape(100,1) # test data for regression
    x_set , y_set = generate_data() # train data for regression
    x_torch , y_torch = torch.Tensor(x_set).to(device), torch.Tensor(y_set).to(device)
    epochs = args.epochs
    batch_size = args.batch_size
    epsilon = 0.01
    # Training an ensemble of 5 networks(MLPS) with MSE
    # Draw Fig1.1
    if args.fig == 1:
        ensemble_layers = 5
        mlps = []
        mlp_optimizers = []
        mlp_criterion = nn.MSELoss()
        for _ in range(ensemble_layers):
            net = MLP(hidden_layers = [100]).to(device)
            mlps.append(net)
            mlp_optimizers.append(torch.optim.Adam(net.parameters(), lr = 0.1))

        for i, net in enumerate(mlps):
            print('Training network ',i+1)
            for epoch in range(epochs):
                mlp_optimizers[i].zero_grad()
                out = net(x_torch)
                mlp_loss = mlp_criterion(y_torch,out)
                if epoch == 0:
                    print('initial loss: ',mlp_loss.item())
                mlp_loss.backward()
                mlp_optimizers[i].step()
            print('final loss: ',mlp_loss.item())
        ys = []
        for net in mlps:
            ys.append(net(torch.FloatTensor(x).to(device)).cpu().detach().numpy())
        ys = np.array(ys)
        mean = np.mean(ys,axis=0)
        std = np.std(ys,axis=0)
        draw_graph(x,x_set,y_set, mean,std)

    # Training a Gaussian MLP(single network) with NLL score rule
    # Draw Fig1.2
    elif args.fig == 2:
        net = GaussianMLP().to(device)
        gaussian_optimizer = torch.optim.Adam(net.parameters(), lr = 0.1)
        for epoch in range(epochs):
            gaussian_optimizer.zero_grad()
            mean, var = net(x_torch)
            gaussian_loss = NLLloss(y_torch, mean, var)
            if epoch == 0:
                print('initial loss: ',gaussian_loss.item())

            gaussian_loss.backward()
            temp = x_torch.grad

            gaussian_optimizer.step()
        print("final loss: ", gaussian_loss.item())
        result_mean, result_var = net(torch.FloatTensor(x).to(device))
        result_mean, result_var = result_mean.cpu().detach().numpy(), result_var.cpu().detach().numpy()
        draw_graph(x,x_set,y_set,result_mean, np.sqrt(result_var))


    #Training a Gaussian MLP with NLL & Adversial Training
    # Draw Fig1.3
    elif args.fig == 3:
        net = GaussianMLP().to(device)
        gaussian_optimizer = torch.optim.Adam(net.parameters(), lr = 0.1)
        x_torch.requires_grad = True
        for epoch in range(epochs):
            gaussian_optimizer.zero_grad()
            mean, var = net(x_torch)
            gaussian_loss = NLLloss(y_torch, mean, var)
            if epoch == 0:
                print('initial loss: ',gaussian_loss.item())

            gaussian_loss.backward(retain_graph = True)
            data_grad = x_torch.grad.data
            sign_grad = data_grad.sign()
            perturbed_data = x_torch + epsilon*sign_grad
            mean_new, var_new = net(perturbed_data)
            gaussian_loss_new = NLLloss(y_torch, mean_new, var_new)
            gaussian_loss += gaussian_loss_new
            net.zero_grad()
            gaussian_loss.backward()
            gaussian_optimizer.step()
        print("final loss: ", gaussian_loss.item())
        result_mean, result_var = net(torch.FloatTensor(x).to(device))
        result_mean, result_var = result_mean.cpu().detach().numpy(), result_var.cpu().detach().numpy()
        draw_graph(x,x_set,y_set,result_mean, np.sqrt(result_var))


    #Training a Gaussian mixture MLP (Deep ensemble) with NLL
    #Draw Fig1.4
    else: #args.fig == 4
        gmm = GaussianMixtureMLP().to(device)
        gmm_optimizers = []
        for i in range(gmm.n_models):
            model = getattr(gmm,'model_'+str(i))
            gmm_optimizers.append(torch.optim.Adam(model.parameters(),lr = 0.1))
        x_torch.requires_grad = True
        for epoch in range(epochs):
            losses =  []
            for i in range(gmm.n_models):
                model = getattr(gmm, 'model_'+str(i))
                loss = train_model_step(model,gmm_optimizers[i],x_torch, y_torch)
                losses.append(loss)
            if epoch == 0:
                print('initial loss: ', losses)

        print("final loss: ", losses)
        result_mean, result_var = gmm(torch.FloatTensor(x).to(device))
        result_mean, result_var = result_mean.cpu().detach().numpy(), result_var.cpu().detach().numpy()
        draw_graph(x,x_set,y_set,result_mean, np.sqrt(result_var))










if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep ensemble')
    parser.add_argument(
        '--epochs',
        type=int,
        default=450)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20)

    parser.add_argument(
        '--fig',
        type=int)
    args = parser.parse_args()

    main(args)
