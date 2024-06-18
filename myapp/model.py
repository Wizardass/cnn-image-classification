import os
import torch

from myapp.model_class import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.ckpt')
        with open(model_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=device)
        self.model = CNN(47)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.label_mapping = {}
        with open('emnist-balanced-mapping.txt', 'r') as f:
            for line in f:
                label, code = line.strip().split()
                self.label_mapping[int(label)] = chr(int(code))

    def predict(self, x):
        self.model.eval()
        x = (x / 255.0) * 2 - 1
        x = x.unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = self.model(x)
        pred_label = output.argmax(dim=1).item()
        pred_char = self.label_mapping[pred_label]
        return pred_char
