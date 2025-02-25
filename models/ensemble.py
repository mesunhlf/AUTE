import torch
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            import random
            # idx = random.randint(1, len(self.models))
            # for i in range(len(self.models)):
            #     if(i == idx):
            #         continue
            #     model = self.models[i]
            #     outputs += F.softmax(model(x), dim=-1)
            # output = outputs / (len(self.models)-1)
            for model in self.models:
                outputs += F.softmax(model(x), dim=-1)
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)     # clip
            return torch.log(output)
        else:
            return self.models[0](x)
