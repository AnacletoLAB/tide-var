import torch.nn as nn

def _initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.BatchNorm1d):
        nn.init.ones_(layer.weight)
        nn.init.zeros_(layer.bias)

class ResNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_groups: int = 9,
        blocks_in_group: int = 2,
        units: int = 256,
        activation_fn: str = 'ReLU',
        dropout: float = 0.0,
        batch_norm: bool = True,
        focal: bool = False,
    ):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_size, units)
        act_fn = getattr(nn, activation_fn)

        self.groups = nn.ModuleList()
        for _ in range(num_groups):
            layers = []
            for _ in range(blocks_in_group):
                layers.append(nn.Linear(units, units))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(units))
                layers.append(act_fn())
                if dropout:
                    layers.append(nn.Dropout(dropout))
            self.groups.append(nn.Sequential(*layers))
        
        self.classifier = nn.Linear(units, num_classes)
        self.apply(_initialize_weights)
        self.focal = focal

    def forward(self, x):
        x = self.input_layer(x)
        for group in self.groups:
            residual = x
            x = group(x)
            x = x + residual
        logits = self.classifier(x)
        if self.focal:
            return logits.squeeze(-1)
        return logits