import numpy as np
import pandas as pd
import scipy as sp
import random
import torch
import copy
import heapq

from recbole.data.dataset import Dataset

class AugmentedDataset(Dataset):


    def __init__(self, config):
        super().__init__(config)
        self.K = config['K']
        self.P = config['P']
    
    def build(self):
        
        self.logger.debug(len(self.inter_feat))

        datasets = super().build()
        test_inter = self.inter_matrix(form='csr')

        self.logger.info(f"Building augmented dataset with K={self.K} and P={self.P}")
        
        self.train_interaction_matrix = datasets[0].inter_matrix(form='csr')
        self.logger.debug(f"Vor augmentation: {len(datasets[0].inter_feat)}")
        datasets[0]._change_feat_format_to_dataframe()


        datasets[0] = self._generate_augmented_train_dataset(datasets[0], K=self.K, P=self.P)
        self.logger.debug("Classname: " + datasets[0].__class__.__name__)

             
        datasets[0]._change_feat_format()
        self.logger.debug("Augmented dataset")
        self.logger.debug(f"Nach augmentation: {len(datasets[0].inter_feat)}")
        return datasets

    def _generate_augmented_train_dataset(self, dataset, K=1, P=15):
            
        
        item_co_occurrence_matrix = self.train_interaction_matrix.transpose().dot(self.train_interaction_matrix)
        degrees_co_occurrence_items = item_co_occurrence_matrix.sum(axis=1)

        similarity_matrix = self._calculate_item_similarity_matrix(item_co_occurrence_matrix, degrees_co_occurrence_items)
        
        augmented_dataset = self._extend_train_dataset(similarity_matrix, dataset, K=K, P=P)

        return augmented_dataset

    

    def _calculate_item_similarity_matrix(self, co_occurrence_matrix, degrees):
        """
        Calculate item-item similarity for all items (optimized).

        Args:
            co_occurrence_matrix (csr_matrix): N x N co-occurrence matrix.
            degrees (np.ndarray): Array of degrees for each item.

        Returns:
            csr_matrix: Matrix of similarities between all items.
        """
        n_items = co_occurrence_matrix.shape[0]
        row_indices, col_indices = co_occurrence_matrix.nonzero()
        data = np.zeros(len(row_indices))
        
        for index, (item_i, item_j) in enumerate(zip(row_indices, col_indices)):
            co_occurrences = co_occurrence_matrix[item_i, item_j]
            degree_i = degrees[item_i]
            degree_j = degrees[item_j]

            # Prevent divide by zero
            if degree_i - co_occurrences != 0 and degree_j != 0:
                similarity_i_j = (co_occurrences / (degree_i - co_occurrences)) * np.sqrt(degree_i / degree_j)
            else:
                similarity_i_j = 0.0

            data[index] = similarity_i_j

        similarities_matrix = sp.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_items, n_items))
        ### maybe not needed
        similarities_matrix = similarities_matrix + similarities_matrix.T  # Fill below the diagonal
        similarities_matrix.setdiag(0)

        return similarities_matrix

    def _extend_train_dataset(self, similarity_matrix, dataset, K=1, P=15):
        
        total_new_items = K * self.user_num
        print("Total new items: " + str(total_new_items))
        augmented_data = []  # List to store data to add to DataFrame

        # Iterate through each user index with interactions
        for user_id in self.user_counter.keys():
            positive_items = self.train_interaction_matrix[user_id].indices

            # Initialize a dictionary to store the K most similar items for each positive item
            top_items_heap = []

            # Find the K most similar items for each positive item
            for item in positive_items:
                # Get the similarity scores of the current item with all other items
                similarity_scores = similarity_matrix[item].toarray().flatten()
                
                # Use a heap to efficiently find the K most similar items
                I_aug = heapq.nlargest(P, range(len(similarity_scores)), key=similarity_scores.__getitem__)
                
                # Store the most similar items
                for similar_item in I_aug:
                    heapq.heappush(top_items_heap, (similarity_matrix[item, similar_item], similar_item))
            
            # Take the top N items from the heap
            top_N_items = heapq.nlargest(K, top_items_heap)
            
            
            # Add the top N items for the current user to the interaction_train_matrix
            for score, item_index in top_N_items:
                self.train_interaction_matrix[user_id, item_index] = 1
            

            for score, item_index in top_N_items:
                augmented_data.append({dataset.uid_field: user_id, dataset.iid_field: item_index})

        print(type(dataset))
        # Add the data to the DataFrame
        dataset.inter_feat = pd.concat([dataset.inter_feat, pd.DataFrame(augmented_data)], ignore_index=True)   
        
        self.logger.debug(dataset.inter_feat.columns)
        
        self.logger.debug(self.train_interaction_matrix.nnz)

        return dataset


    def _change_feat_format_to_dataframe(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`."""
        for feat_name in self.feat_name_list:
            feat_df = getattr(self, feat_name)
            feat_interaction = self._interaction_to_dataframe(feat_df)
            setattr(self, feat_name, feat_interaction)

    def _interaction_to_dataframe(self, interaction):
        """Convert :class:`~recbole.data.interaction.Interaction` to :class:`pandas.DataFrame`.

        Args:
            interaction (:class:`~recbole.data.interaction.Interaction`): interaction data to be converted.

        Returns:
            pandas.DataFrame: Converted data.
        """
        data_dict = interaction.numpy()  # Convert interaction to a dictionary of numpy arrays
        data = {}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.numpy()  # Convert torch tensors to numpy arrays
            data[k] = v

        # Create a DataFrame using the converted dictionary
        df = pd.DataFrame(data)

        return df