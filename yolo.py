from torch import nn

def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.LeakyReLU(negative_slope=0.1)
    ]

class YOLO(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.layer = nn.Sequential(
            # layer1
            *conv_layer(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer2
            *conv_layer(in_channels=64, out_channels=192, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer3
            *conv_layer(in_channels=192, out_channels=128, kernel_size=1),
            *conv_layer(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            *conv_layer(in_channels=256, out_channels=256, kernel_size=1),
            *conv_layer(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),   
            # layer 4
            *conv_layer(in_channels=512, out_channels=256, kernel_size=1),
            *conv_layer(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            *conv_layer(in_channels=512, out_channels=256, kernel_size=1),
            *conv_layer(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            *conv_layer(in_channels=512, out_channels=256, kernel_size=1),
            *conv_layer(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            *conv_layer(in_channels=512, out_channels=256, kernel_size=1),
            *conv_layer(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            *conv_layer(in_channels=512, out_channels=512, kernel_size=1),
            *conv_layer(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # layer 5
            *conv_layer(in_channels=1024, out_channels=512, kernel_size=1),
            *conv_layer(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            *conv_layer(in_channels=1024, out_channels=512, kernel_size=1),
            *conv_layer(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            *conv_layer(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            *conv_layer(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            # layer 6
            *conv_layer(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            *conv_layer(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            # layer 7
            nn.Flatten(),
            nn.Linear(in_features=self.S * self.S * 1024, out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=4096, out_features=self.S * self.S * (self.B * 5 + self.C)),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        o = self.layer(x)
        o = o.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        grid_probs = o[:, :, :, -self.C:]
        boxes = o[:, :, :, [i for i in range(self.B * 5) if i % 5 != 0]].reshape(batch_size, self.S, self.S, -1, 4)
        boxes_confidences = o[:, :, :, [i for i in range(self.B * 5) if i % 5 == 0]]
        return boxes, boxes_confidences, grid_probs