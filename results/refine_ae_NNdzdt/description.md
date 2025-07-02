Looking for better training performance on the black-box NN version of dz/dt by refining the network structure and/or initialization.

## Base case:
10 epochs, (40, 40, 40) layer size; see f7a82c

![title](box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_f7a82c58d1114b5fa0a41b19b8e6c7cb/box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_f7a82c58d1114b5fa0a41b19b8e6c7cb_predictions.png)
![title](box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_f7a82c58d1114b5fa0a41b19b8e6c7cb/box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_f7a82c58d1114b5fa0a41b19b8e6c7cb_trajectories.png)
```aiignore
class NNDerivatives(torch.nn.Module):
    def __init__(self, n_latent=3, layer_size=None):
        super(NNDerivatives, self).__init__()
        self.n_latent = n_latent
        if layer_size is None:
            layer_size = (n_latent, n_latent, n_latent)
        else:
            assert len(layer_size) == 3

        self.layer1 = Linear(n_latent, layer_size[0])
        self.layer2 = Linear(layer_size[0], layer_size[1])
        self.layer3 = Linear(layer_size[1], layer_size[2])
        self.layer4 = Linear(layer_size[2], n_latent)
        self.activation1 = ELU()
        self.activation2 = ELU()
        self.activation3 = ELU()
        self.activation4 = Tanh()

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [
            self.activation1,
            self.activation2,
            self.activation3,
            self.activation4,
        ]

        self.apply(self.init_weights)

    def forward(self, z, M=None):
        if M is not None:
            x = torch.cat([z, M], dim=-1)
        else:
            x = z
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
        for i, layer in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)

    def set_weights(self, weights, biases):
        for i, layer in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
```

## 1. Removing tanh final layer
No activation4; 4 linear layers and 3 activation; a20629
![title](box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_a20629bbfcba419990908aaa2bfdd72e/box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_a20629bbfcba419990908aaa2bfdd72e_predictions.png)
![title](box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_a20629bbfcba419990908aaa2bfdd72e/box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_a20629bbfcba419990908aaa2bfdd72e_trajectories.png)

## 2. Switching ELU to SiLU on hidden layers
Still no activation 4; change hidden layers to SiLU; b5198f35

## 3. GOLDEN TICKET: Use smaller initialization for the final output layer & kaiming normal elsewhere
8c4819

![hi](box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_8c4819d73000425d97f88022b1c2a22c/box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_8c4819d73000425d97f88022b1c2a22c_predictions.png)
![hi](box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_8c4819d73000425d97f88022b1c2a22c/box_FFNN_latent3_layers(40,%2040,%2040)_tr10_lr0.001_bs32_weights1.0-8.537322998046875-853.7322998046875_8c4819d73000425d97f88022b1c2a22c_trajectories.png)

See also:
- fc335c for 50 epoch training version
- erf_ i.e. XXXXX for 50 epoch erf version
```aiignore
class NNDerivatives(torch.nn.Module):
    def __init__(self, n_latent=3, layer_size=None):
        super(NNDerivatives, self).__init__()
        self.n_latent = n_latent
        if layer_size is None:
            layer_size = (n_latent, n_latent, n_latent)
        else:
            assert len(layer_size) == 3

        self.layer1 = Linear(n_latent, layer_size[0])
        self.layer2 = Linear(layer_size[0], layer_size[1])
        self.layer3 = Linear(layer_size[1], layer_size[2])
        self.layer4 = Linear(layer_size[2], n_latent)
        self.activation1 = SiLU()
        self.activation2 = SiLU()
        self.activation3 = SiLU()

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.act = [
            self.activation1,
            self.activation2,
            self.activation3,
        ]

        self.initialize_network()

    def forward(self, z, M=None):
        if M is not None:
            x = torch.cat([z, M], dim=-1)
        else:
            x = z
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)

        return x

    def get_weights(self):
        weights = []
        biases = []
        for i, layer in enumerate(self.layers):
            weights.append(layer.weight)
            biases.append(layer.bias)

        return (weights, biases)

    def set_weights(self, weights, biases):
        for i, layer in enumerate(self.layers):
            layer.weight.data = weights[i]
            layer.bias.data = biases[i]

    def initialize_network(self):
        for i, module in enumerate(self.layers):
            if isinstance(module, nn.Linear):
                if i < len(self.layers) - 1:  # Hidden layers with SiLU
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0.0)
                else:  # Output layer (no activation)
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    nn.init.constant_(module.bias, 0.0)
```
