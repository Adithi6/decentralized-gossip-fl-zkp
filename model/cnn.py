import logging
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        conv1_channels: int,
        conv2_channels: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        flattened_dim = self._get_flattened_dim(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        logging.info(
            "SmallCNN model initialized | "
            f"input_channels={input_channels} "
            f"input_size=({input_height}, {input_width}) "
            f"conv1_channels={conv1_channels} "
            f"conv2_channels={conv2_channels} "
            f"hidden_dim={hidden_dim} "
            f"num_classes={num_classes} "
            f"flattened_dim={flattened_dim}"
        )

    def _get_flattened_dim(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
    ) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            out = self.features(dummy)
            return out.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x