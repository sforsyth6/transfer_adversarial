import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm


# adversarial attack class
class Attacks:

    def __init__(self,
                 steps: int,
                 gamma: float =0.05,
                 alpha: float =1e4,
                 epsilon: float=0.1,
                 init_norm: float = 1.,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu'),
                 loss_fxn = nn.CrossEntropyLoss()):

        self.steps = steps
        self.gamma = gamma
        self.alpha = alpha
        self.init_norm = init_norm

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm

        self.device = device
        self.loss_fxn = loss_fxn

    # basic projected gradient descent attack
    def attack_pgd(self, model: nn.Module, inputs: torch.Tensor, attack_fraction=0.5) -> torch.Tensor:
        cuda = torch.cuda.is_available()

        batched_deltas = []

        for batch_idx, (data, target) in enumerate(tqdm(inputs, desc="Adv")):

            """ Construct FGSM adversarial examples on the examples X"""
            delta = torch.zeros_like(data, requires_grad=True)

            if cuda:
                delta = delta.cuda()


            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            for t in range(self.steps):

                adv_data = data[0] + delta



                output = model(adv_data)

                loss = self.loss_fxn(output, target)

                loss.backward()

                delta.grad.data = (delta + data[0].shape[0]*self.alpha*delta.grad.data).clamp(-self.epsilon,self.epsilon)
                delta.grad.zero_()

            batched_deltas.append(delta.detach())


        return torch.mean(torch.stack(batched_deltas))


    # Projected Gradient Descent.
    def pgd_attack(self,model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, loss_criterion, epsilon, radius):
        # Perturbation.
        pgd = inputs.new_zeros(inputs.shape)

        # We want the gradients.
        pgd.requires_grad = True
        inputs.requires_grad = True

        # Compute perturbed input.
        x_pgd = inputs + pgd

        for _ in range(self.steps):
            # Compute data gradient.
            x_pgd.retain_grad()  # Gradient is accumulated only in leaf Variables usually.
            predictions = model(x_pgd)
            loss = loss_criterion(predictions, targets)
            loss.requires_grad = True
            model.zero_grad()
            # loss.backward(create_graph=True)
            loss.backward()
            print(x_pgd)
            print(x_pgd.requires_grad) # returns False 
            x_pgd_grad = x_pgd.grad
            print(x_pgd_grad)
            print(type(x_pgd_grad))
            # Collect the element-wise sign of the data gradient.
            sign_x_pgd_grad = x_pgd_grad.sign()
            # Create the perturbed input.
            pgd = pgd + epsilon*sign_x_pgd_grad
            # Clip to radius.
            pgd = torch.clamp(pgd, -radius, radius)
            # Update perturbed input.
            x_pgd = inputs + pgd
        # Return the perturbed input.
        return x_pgd
