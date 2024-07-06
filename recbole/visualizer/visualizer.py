import os
import json
import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger

class PlotVisualizer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.logger = getLogger()
        self.logger.setLevel("WARNING")
        self.colors = ['#000000', '#CCFF00', '#4C7D8A', '#E57050', '#813353']
        plt.rcParams.update({'font.size': 12})

    def read_data(self):
        # Collect all json data from the specified directory
        data = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r') as file:
                    file_data = json.load(file)
                    data.update(file_data)
        return data

    def plot_data(self):
        data = self.read_data()

        # Prepare plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300) 

        lines = []
        labels = []

        for i, (key, value) in enumerate(data.items()):
            epochs = list(map(int, value['train_loss'].keys()))
            train_loss = list(value['train_loss'].values())
            recall = list(value['recall@20'].values())

            # Plot Training Loss
            line1, = ax1.plot(epochs, train_loss, label=key, color=self.colors[i])
            lines.append(line1)
            labels.append(key)

            # Plot Recall@20
            line2, = ax2.plot(epochs, recall, label=key, color=self.colors[i])
            # Ensure we don't add the same label again for the unified legend
        
        labels, lines = zip(*sorted(zip(labels, lines), key=lambda x: x[0]))

        # Set labels for x and y axes
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Recall@20')
        
        # Create a unified legend at the top of the figure
        fig.legend(lines, labels, loc='upper center', ncol=len(labels))
        fig.tight_layout(pad=3)

        plt.savefig(os.path.join(self.data_dir, 'Ablation Study Movielens 100k.png'), dpi=300)

if __name__ == '__main__':
    visualizer = PlotVisualizer(data_dir='log/plotting_data_100k_ablation study')
    visualizer.plot_data()
