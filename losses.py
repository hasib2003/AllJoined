import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# Thanks to: https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(config.device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).to(config.device).float())
            
    # def forward(self, emb_i, emb_j):
    def forward(self, z_i, z_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # Already getting normalize vectors
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)

        representations   = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return torch.mean(loss)
    




class ContrastiveDynamicLoss(nn.Module):
    def __init__(self):
        super(ContrastiveDynamicLoss, self).__init__()

    def category_based_sim(self, categories):
        anchors = categories[:, 0, :]
        positives = categories[:, 1, :]
        negatives = categories[:, 2, :]

        p_sims = torch.sum(anchors == positives, dim=1) / anchors.shape[-1]
        n_sims = torch.sum(anchors == negatives, dim=1) / anchors.shape[-1]

        return p_sims, n_sims

    def forward(self, embeds, categories):
        """
        embeds: Tensor of shape (num_samples, 3, emb_dim)
        categories: Tensor of shape (num_samples, 3, num_categories)
        """

        assert embeds.shape[:2] == categories.shape[:2], "Shapes of embeds and categories must match"

        # Calculate margins based on category similarity
        margins_pos, margins_neg = self.category_based_sim(categories)

        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-08)

        anc = embeds[:, 0, :]
        pos = embeds[:, 1, :]
        neg = embeds[:, 2, :]

        sim_pos = cosine_sim(anc, pos)
        sim_neg = cosine_sim(anc, neg)

        loss_pos = torch.abs(sim_pos - margins_pos)
        loss_neg = torch.abs(sim_neg - margins_neg)

        # Calculate the total loss
        loss = torch.mean(loss_pos + loss_neg)

        return loss
