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
    
    def read_json(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data

    def plot_parameter_study(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                data = self.read_json(filepath)
                self.plot_parameter_data(data, filename)

    def plot_parameter_data(self, data, filename):
        def extract_value(key):
            try:
                return int(key.split(':')[1])
            except ValueError:
                return float(key.split(':')[1])

        q_values = sorted(data.keys(), key=lambda x: extract_value(x))
        q_labels = [extract_value(q) for q in q_values]
        param_name = q_values[0].split(':')[0]  # Extract the parameter name (e.g., "Q" or "K")
        recall_values = [data[q]['recall@20'] for q in q_values]
        ndcg_values = [data[q]['ndcg@20'] for q in q_values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        
        if isinstance(q_labels[0], int):
            q_indices = range(len(q_values))  # Create an index range for even spacing
            xticks = q_indices
        else:
            q_indices = range(len(q_values))  # Create an index range for float labels
            xticks = q_labels
        
        # Plot Recall@20
        ax1.plot(q_indices, recall_values, marker='o', linestyle='-', color='b')
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('Recall@20')
        if xticks is not None:
            ax1.set_xticks(q_indices)  # Set x-ticks to the index range
            ax1.set_xticklabels(q_labels)
        ax1.grid(True)
        
        # Plot NDCG@20
        ax2.plot(q_indices, ndcg_values, marker='o', linestyle='-', color='g')
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('NDCG@20')
        if xticks is not None:
            ax2.set_xticks(q_indices)  # Set x-ticks to the index range
            ax2.set_xticklabels(q_labels)
        ax2.grid(True)
        
        # Save and show the plot
        plot_filename = f'{filename.split(".")[0]}_study_plot.png'
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.data_dir, plot_filename), dpi=300)

if __name__ == '__main__':
    #visualizer = PlotVisualizer(data_dir='log/plotting_data/plotting_data_100k_ablation study')
    #visualizer.plot_data()
    visualizer = PlotVisualizer(data_dir='log/plotting_data/parameter_study')
    visualizer.plot_parameter_study()
