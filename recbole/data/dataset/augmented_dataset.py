import numpy as np
import pandas as pd
import scipy as sp
import random
import torch
import copy

from recbole.data.dataset import Dataset

class AugmentedDataset(Dataset):


    def __init__(self, config):
        super().__init__(config)
    
    def build(self):
        
        self.logger.debug(len(self.inter_feat))

        #inter_mat = self.inter_feat.inter_matrix(form='csr')
        # for idx, item in enumerate(self.inter_feat.values):
        #     self.logger.debug(item)
        #     if idx == 10:
        #         break
        datasets = super().build()
        test_inter = self.inter_matrix(form='csr')
        self.logger.debug(test_inter.nnz)

        self._change_feat_format_to_dataframe()

        
        self.train_interaction_matrix = datasets[0].inter_matrix(form='csr')
        #nnz_entierr = self.train_interaction_matrix[2].nonzero()[1]
        #self.logger.debug(nnz_entierr)

        # for idx, item in enumerate(self.train_interaction_matrix):
        #     self.logger.debug(item)
        #     if idx == 10:
        #         break

        self._generate_augmented_train_dataset()
        self.logger.debug("Classname: " + datasets[0].__class__.__name__)

             
        self._change_feat_format()
        self.logger.debug("Augmented dataset")
        
        return datasets

    def _generate_augmented_train_dataset(self, K_si=1):
            
        
        item_co_occurrence_matrix = self.train_interaction_matrix.transpose().dot(self.train_interaction_matrix)
        degrees_co_occurrence_items = item_co_occurrence_matrix.sum(axis=1)

        similarity_matrix = self._calculate_item_similarity_matrix(item_co_occurrence_matrix, degrees_co_occurrence_items)
        
        self._extend_train_dataset(similarity_matrix)

        return

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

        return similarities_matrix

    def _extend_train_dataset(self, similarity_matrix, N=1):
        total_new_items = N * self.user_num
        print("Total new items: " + str(total_new_items))
        # Initialize an empty matrix to store the extended interaction data
        new_interaction_data = np.zeros((self.user_num, self.item_num + total_new_items))

        # Iterate through each user index with interactions
        for user_id in self.user_counter.keys():
            positive_items = self.train_interaction_matrix[user_id].indices
            self.logger.debug("Positive items: " + str(positive_items))
        return


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