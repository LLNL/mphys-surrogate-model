import numpy as np
import torch
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import Linear, ReLU, Sigmoid, ConstantPad1d, Identity

"""
Convolutional NN Autoencoder; can operate on multiple channels of input (such as number and mass densities)
"""
class CNNEncoderVAE(torch.nn.Module):
    def __init__(self,n_channels=2,n_bins=35,n_latent=10):
        super(CNNEncoderVAE, self).__init__()
        self.n_bins = n_bins
        self.conv1 = Conv1d(in_channels=n_channels,out_channels=n_channels*2,kernel_size=4,stride=2,padding=1)
        self.activation1 = ReLU()
        self.conv2 = Conv1d(in_channels=n_channels*2,out_channels=n_channels*4,kernel_size=4,stride=2,padding=1)
        self.activation2 = ReLU()
        self.conv3 = Conv1d(in_channels=n_channels*4,out_channels=n_channels*2,kernel_size=4,stride=2,padding=1)
        self.activation3 = ReLU()
        self.lin1 = Linear(30, n_latent) #Lin(48,n_latent)

        self.layer_id = ["conv1", "conv2", "conv3", "lin1"]
        self.layers = [self.conv1, self.conv2, self.conv3, self.lin1]
        
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)

    def forward(self,x):
        n_bins = self.n_bins
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = x.view(-1,30) #48)
        x = self.lin1(x)

        return x
    
    def get_weights(self):
        weights = {}
        biases = {}
        for (i, layer) in enumerate(self.layers):
            weights[self.layer_id[i]] = layer.weight
            biases[self.layer_id[i]] = layer.bias

        return (weights, biases)
    
    def set_weights(self, weights, biases):
        for (i, layer) in enumerate(self.layers):
            layer.weight.data = weights[self.layer_id[i]]
            layer.bias.data = biases[self.layer_id[i]]

class CNNDecoder(torch.nn.Module):
    def __init__(self,n_channels=2,n_bins=100,n_latent=10,n_hidden=50):
        super(CNNDecoder, self).__init__()

        self.n_latent = n_latent
        self.n_channels = n_channels

        self.n_bins = n_bins
        self.lin = Linear(n_latent, 30) #Lin(n_latent,48)
        self.conv1 = ConvTranspose1d(in_channels=n_channels*2,out_channels=n_channels*4,kernel_size=4,stride=2,padding=1)
        self.activation1 = ReLU()
        self.constantpad1d1 = ConstantPad1d((1,0),0)
        self.conv2 = ConvTranspose1d(in_channels=n_channels*4,out_channels=n_channels*2,kernel_size=4,stride=2,padding=1)
        self.activation2 = ReLU()
        self.conv3 = ConvTranspose1d(in_channels=n_channels*2,out_channels=n_channels,kernel_size=4,stride=2,padding=1)
        self.activation3 = ReLU()
        self.lin2 = Linear(n_bins,n_bins)
        self.activation4 = Sigmoid()

        self.layer_id = ["lin", "conv1", "conv2", "conv3", "lin2"]
        self.layers = [self.lin, self.conv1, self.conv2, self.conv3, self.lin2]
        
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        
    def forward(self,x):
        inp = x
        x = self.lin(inp)
        x = x.reshape(-1, 2, 15) #x.reshape(-1,4,12)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.constantpad1d1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.constantpad1d1(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.constantpad1d1(x)
        x = self.lin2(x)
        x = self.activation4(x) 

        return x
    
    def get_weights(self):
        weights = {}
        biases = {}
        for (i, layer) in enumerate(self.layers):
            weights[self.layer_id[i]] = layer.weight
            biases[self.layer_id[i]] = layer.bias

        return (weights, biases)
    
    def set_weights(self, weights, biases):
        for (i, layer) in enumerate(self.layers):
            layer.weight.data = weights[self.layer_id[i]]
            layer.bias.data = biases[self.layer_id[i]]

class CNNAutoEncoder(torch.nn.Module):
    def __init__(self,n_channels=2,n_bins=100,n_latent=10):
        super(CNNAutoEncoder, self).__init__()

        self.encoder = CNNEncoderVAE(n_channels=n_channels,n_bins=n_bins,n_latent=n_latent)
        self.decoder = CNNDecoder(n_channels=n_channels,n_bins=n_bins,n_latent=n_latent)

    def forward(self,x):

        latent = self.encoder(x)

        reconstruction = self.decoder(latent) 

        return reconstruction


"""
Feed-forward neural network autoencoder
"""
class FFNNEncoderVAE(torch.nn.Module):
    def __init__(self,n_bins=127,n_latent=3):
        super(FFNNEncoderVAE, self).__init__()
        self.n_bins = n_bins
        self.layer1 = Linear(n_bins, int(n_bins / 2))
        self.activation1 = ReLU()
        self.layer2 = Linear(int(n_bins / 2), int(n_bins / 4))
        self.activation2 = ReLU()
        self.layer3 = Linear(int(n_bins / 4), int(n_bins / 8))
        self.activation3 = ReLU()
        self.layer4 = Linear(int(n_bins / 8), n_latent)
        self.activation4 = Identity()

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [self.activation1, self.activation2, self.activation3, self.activation4]

    def forward(self,x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)
        
        return x
    
    def get_weights(self):
        weights = []
        biases = []
        for (i, layer) in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)
    
    def set_weights(self, weights, biases):
        for (i, layer) in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

class FFNNDecoder(torch.nn.Module):
    def __init__(self,n_bins=128,n_latent=3):
        super(FFNNDecoder, self).__init__()

        self.n_bins = n_bins
        self.layer1 = Linear(n_latent, int(n_bins / 8))
        self.layer2 = Linear(int(n_bins / 8), int(n_bins / 4))
        self.layer3 = Linear(int(n_bins / 4), int(n_bins / 2))
        self.layer4 = Linear(int(n_bins / 2), n_bins)
        self.activation1 = ReLU()
        self.activation2 = ReLU()
        self.activation3 = ReLU()
        self.activation4 = Sigmoid()

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [self.activation1, self.activation2, self.activation3, self.activation4]
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)
        x = self.activation4(x)
        
        return x
    
    def get_weights(self):
        weights = []
        biases = []
        for (i, layer) in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)
    
    def set_weights(self, weights, biases):
        for (i, layer) in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

class FFNNAutoEncoder(torch.nn.Module):
    def __init__(self,n_bins=100,n_latent=10):
        super(FFNNAutoEncoder, self).__init__()

        self.encoder = FFNNEncoderVAE(n_bins=n_bins,n_latent=n_latent)
        self.decoder = FFNNDecoder(n_bins=n_bins,n_latent=n_latent)

        self.initialize_weights()

    def forward(self,x):
        
        latent = self.encoder(x)
        reconstruction = self.decoder(latent) 

        return reconstruction
    
    def initialize_weights(self):
        for network in (self.encoder, self.decoder):
            for layer in network.layers:
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                torch.nn.init.uniform_(layer.bias)
    
"""
Utility functions
"""
def get_latent_var(model, dataloader, device, n_latent):
    dataset = dataloader.dataset
    latents = np.zeros((len(dataset), n_latent))

    jj=0
    for data in dataloader:
        bin0 = data
        bin0 = bin0.to(device)
        latent = model.encoder(bin0.float())
        bs = latent.shape[0]

        latents[jj:jj+bs,:]=latent.detach().cpu().numpy().reshape(bs, n_latent)
        jj+=bs
    
    return latents


