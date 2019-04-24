import torchvision.models as models
from torch import nn
class EncoderCNN(nn.Module):
    def __init__(self, embed_size,model_type="resenet-50"):
        """Load the pretrained Net and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = getattr(models,model_type)(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features