import numpy as np
import torch
from collections import Counter
from recbole.sampler import Sampler
from recbole.data.interaction import Interaction, cat_interactions
from logging import getLogger


class PSST(Sampler):
    """
    A sampler for generating positive interactions for recommendation models.
    This is the PSST component of the GCN-AST model.

    Args:
        phases (list): List of phases for which the sampler is used.
        datasets (list): List of datasets for each phase.
        distribution (str, optional): Distribution of sampling. Defaults to "uniform".
        alpha (float, optional): Parameter for distribution. Defaults to 1.0.
    """

    def __init__(self, phases, datasets, distribution="uniform", alpha=1.0):
        super().__init__(phases, datasets, distribution, alpha)
        self.logger = getLogger()

    def predict_for_unused_ids(self, top_k):
        """
        Predicts top-K items for users based on unused item IDs.

        Args:
            top_k (int): Number of top items to predict.

        Returns:
            dict: Dictionary containing the top-K item IDs and their scores for each user.
        """
        # Get used and unused item IDs
        used_item_id, unused_item_id = self.get_used_and_unused_ids()
        
        # set phase
        if self.phase == "train":
            unused_item_ids_for_phase = unused_item_id[self.phase]
        else:
            raise AssertionError("Only support predict sampling for train phase")
        
        results = {}  # Dictionary to store the predictions
        self.logger.debug(f"Predicting for {len(unused_item_ids_for_phase)} users")
        for uid, unused_ids in enumerate(unused_item_ids_for_phase):
            if not unused_ids or uid == 0:
                continue  # Skip if there are no unused IDs for the user
            unused_ids = list(unused_ids)
            self.logger.debug(f"Predicting for user {uid} with {len(unused_ids)} unused items")
            # Create interaction for predict function
            interaction = Interaction({self.uid_field: torch.tensor([uid] * len(unused_ids)).to(self.model.device), self.iid_field: torch.tensor(unused_ids).to(self.model.device)})
            
            # Call the predict function
            with torch.no_grad():
                scores = self.model.predict(interaction)
                
                if self.model.__class__.__name__ == "ANS":
                    scores = scores.sum(dim=1)
                else:
                    pass
                    
            
            # Get the top-K items and their scores
            top_k_indices = torch.topk(scores, top_k).indices
            top_k_item_ids = interaction[self.iid_field][top_k_indices]
            self.logger.debug(f"Top {top_k} items for user {uid}: {top_k_item_ids}")
            top_k_scores = scores[top_k_indices]
            # Store the results
            results[uid] = {
                "item_ids": top_k_item_ids.tolist(),
                "scores": top_k_scores.tolist()
            }
        
        return results

    def get_top_k_interactions(self, top_k):
        """
        Gets the top-K predicted interactions for each user.

        Args:
            top_k (int): Number of top interactions to retrieve.

        Returns:
            Interaction: Concatenated interactions containing the top-K predicted interactions for all users.
        """
        top_k_predictions = self.predict_for_unused_ids(top_k)
        interactions = []
        for uid, data in top_k_predictions.items():
            item_ids = data["item_ids"]
            interaction = Interaction({self.uid_field: torch.tensor([uid] * len(item_ids)), self.iid_field: torch.tensor(item_ids)})
            interactions.append(interaction)
        
        interactions = cat_interactions(interactions)
        return interactions
    
    def add_potential_positives_to_dataset(self, top_k):
        """
        Adds potential positive interactions to the dataset.

        Args:
            top_k (int): Number of top interactions to add.

        Returns:
            Interaction: Updated interaction feature matrix with the added potential positive interactions.
        """
        self.model.eval()
        interactions = self.get_top_k_interactions(top_k)
        self.logger.debug(f"Adding {len(interactions)} potential positive interactions to the dataset")
        self.logger.debug(self.datasets[0].inter_feat.columns)
        new_inter_feat = cat_interactions([self.datasets[0].inter_feat, interactions])
        self.datasets[0].inter_feat = new_inter_feat
        self.model.train()
        return new_inter_feat

    def get_used_and_unused_ids(self):
        """
        Returns the used and unused item IDs.

        Returns:
            tuple: (used_ids, unused_ids)
            - used_ids (dict): Used item_ids is the same as positive item_ids.
              Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
            - unused_ids (dict): Unused item_ids. Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
        """
        used_item_id = dict()
        unused_item_id = dict()
        all_items = set(range(1,self.item_num))  # All possible item IDs
        last = [set() for _ in range(self.user_num)]
        
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            for uid, iid in zip(
                dataset.inter_feat[self.uid_field].numpy(),
                dataset.inter_feat[self.iid_field].numpy(),
            ):
                cur[uid].add(iid)
            last = used_item_id[phase] = cur
            unused_item_id[phase] = np.array([all_items - user_set for user_set in cur])

        for used_item_set in used_item_id[self.phases[-1]]:
            if len(used_item_set) + 1 == self.item_num:  # [pad] is an item.
                raise ValueError(
                    "Some users have interacted with all items, "
                    "which we cannot sample negative items for them. "
                    "Please set `user_inter_num_interval` to filter those users."
                )
        
        return used_item_id, unused_item_id
    
    def set_model(self, model):
        """
        Sets the recommendation model.

        Args:
            model: The recommendation model to be set.
        """
        self.model = model

    def update_sampler_inter_feat(self, inter_feat):
        """
        Updates the interaction feature matrix of the sampler.

        Args:
            inter_feat: The updated interaction feature matrix.
        """
        self.datasets[0].inter_feat = inter_feat
        self.uid_field = self.datasets[0].uid_field
        self.iid_field = self.datasets[0].iid_field

        self.user_num = self.datasets[0].user_num
        self.item_num = self.datasets[0].item_num
        self.used_ids = self.get_used_ids()
        self.used_ids = self.used_ids["train"]