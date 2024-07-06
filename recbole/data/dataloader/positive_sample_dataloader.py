import torch
from torch.utils.data import DataLoader
from recbole.data.interaction import Interaction, cat_interactions

class PredictiveSampleDataLoader(DataLoader):
    """A DataLoader that predicts the best n items for users and adds them as new positive samples."""

    def __init__(self, config, dataset, model, sampler, top_n=5, shuffle=False):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.sampler = sampler
        self.top_n = top_n
        self.shuffle = shuffle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize DataLoader
        super().__init__(dataset=list(range(len(dataset))),
                         batch_size=config["train_batch_size"],
                         collate_fn=self.collate_fn,
                         shuffle=shuffle,
                         num_workers=config["worker"],
                         generator=torch.Generator().manual_seed(config["seed"]))

    def collate_fn(self, indices):
        """Collate function to add predicted items as positive samples."""
        inter_feat = self.dataset[indices]
        new_inter_feat = self.predict_and_add_positive_samples(inter_feat)
        return new_inter_feat

    def predict_and_add_positive_samples(self, inter_feat):
        """Predict the top n items for each user and add them as new positive samples."""
        user_ids = inter_feat[self._dataset.uid_field].to(self.model.device)
        item_ids = inter_feat[self._dataset.iid_field].to(self.model.device)
        
        # Create a mask to avoid predicting the same items again
        mask = torch.ones((len(user_ids), len(item_ids)), device=self.model.device)
        for i, user_id in enumerate(user_ids):
            mask[i, item_ids == item_ids[i]] = 0
        
        # Predict scores for all items for each user
        all_item_ids = torch.arange(self._dataset.item_num).to(self.model.device)
        all_scores = []
        for user_id in user_ids:
            user_tensor = torch.full_like(all_item_ids, user_id, device=self.model.device)
            scores = self.model.predict(Interaction({self._dataset.uid_field: user_tensor, self._dataset.iid_field: all_item_ids}))
            all_scores.append(scores)
        
        all_scores = torch.stack(all_scores, dim=0)
        all_scores = all_scores.masked_fill(mask == 0, float('-inf'))
        
        # Get top n items for each user
        _, top_n_indices = torch.topk(all_scores, self.top_n, dim=1)
        new_item_ids = all_item_ids[top_n_indices]
        
        # Create new positive interactions
        new_interactions = []
        for i, user_id in enumerate(user_ids):
            for item_id in new_item_ids[i]:
                new_interactions.append({self._dataset.uid_field: user_id.item(), self._dataset.iid_field: item_id.item()})
        
        new_inter_feat = Interaction(new_interactions).to(self.model.device)
        combined_inter_feat = cat_interactions([inter_feat, new_inter_feat])
        
        return combined_inter_feat
