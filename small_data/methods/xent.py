import torch
from torch import nn
from typing import Callable, Tuple

from .common import BasicAugmentation


class CrossEntropyClassifier(BasicAugmentation):
    """ Standard cross-entropy classification as baseline.

    See `BasicAugmentation` for a documentation of the available hyper-parameters.
    """

    def get_loss_function(self) -> Callable:
        """ Returns a loss function.
		
		Returns
		------
		callable(*outputs, targets)
			A loss function taking as arguments all model outputs followed by
			target class labels and returning a single loss value.
        """
        return nn.CrossEntropyLoss(reduction='mean') # xent is default

    def get_optimizer(self, 
        model: nn.Module, 
        max_epochs: int, 
        max_iter: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """ Instantiates an optimizer and learning rate schedule.

        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        max_epochs : int
            The total number of epochs.
        max_iter : int
            The total number of iterations (epochs * batches_per_epoch).
        
        Returns
        -------
        optimizer : torch.optim.Optimizer
        lr_schedule : torch.optim.lr_scheduler._LRScheduler
        """
        optimizer = torch.optim.SGD(model.parameters(), 
			lr=self.hparams['lr'], 
			momentum=self.hparams['momentum'], 
			weight_decay=self.hparams['weight_decay']
		)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        return optimizer, scheduler
        
    @staticmethod
    def default_hparams() -> dict:
        """
        default -> { 'lr' : 0.01, 'weight_decay' : 0.001 }
        """
        hparams_dict = {
            **super(CrossEntropyClassifier, CrossEntropyClassifier).default_hparams(),
            'momentum' : 0.9
        }
        return hparams_dict
