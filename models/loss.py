import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    """
    Computes nll loss (Eq. (6)), coverage loss (Eq. (12)),
    and the composite loss function that combines the two (Eq. (13)).
    """
    def __init__(self, args):
        super().__init__()
        self.use_coverage = args.use_coverage
        self.cov_weight = args.cov_weight   # hyperparameter lambda in Eq. (13)
        self.pad_id = args.pad_id

    def nll_loss(self, output, target):
        """
        Negative log likelihood of target word - Eq. (6)
        Args:
            output: predicted probs from each timestep      [B x V_x T]
            target: answer ids using extended vocab         [B x T]

        Returns:
            loss: nll loss value; averaged over batch & timestep
        """
        loss = F.nll_loss(output, target,
                          ignore_index=self.pad_id,
                          reduction='mean')
        return loss

    def cov_loss(self, attn_dist, coverage, dec_pad_mask, dec_len):
        """
        Coverage loss at timestep t - Eq. (12)
        Args:
            attn_dist: attention distribution from all timesteps            [B x L x T]
            coverage: sum of previous attn dist's from all timesteps        [B x L x T]
            dec_pad_mask: target sequence padding masks [PAD] -> True       [B x T]
            dec_len: target sequence lengths                                [B]

        Returns:
            loss: coverage loss value; averaged over batch & timestep
        """
        min_val = torch.min(attn_dist, coverage)    # [B x L x T]
        loss = torch.sum(min_val, dim=1)            # [B x T]

        # ignore loss from [PAD] tokens
        loss = loss.masked_fill_(
            dec_pad_mask,
            0.0
        )
        avg_loss = torch.sum(loss) / torch.sum(dec_len)
        return avg_loss

    def forward(self, output, batch):
        """
        Eq. (13) - Composite loss
        Args:
            output: a dictionary of model outputs with the following keys
                - final_dist
                - attn_dist
                - coverage
            batch: `Batch` instance

        Returns:
            loss: final composite loss value
        """
        final_dist = output['final_dist']
        dec_target = batch.dec_target
        loss = self.nll_loss(output=final_dist, target=dec_target)

        if self.use_coverage:
            attn_dist = output['attn_dist']
            coverage = output['coverage']
            dec_pad_mask = batch.dec_pad_mask
            dec_len = batch.dec_len
            cov = self.cov_loss(attn_dist, coverage, dec_pad_mask, dec_len)
            loss += self.cov_weight * cov

        return loss


def build_criterion(config):
    criterion = Loss(args=config.loss.args)
    return criterion