import torch
from torch.optim.optimizer import Optimizer


class SGDOptimizer(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self, closure):
        loss = closure()
        loss.backward()

        with torch.no_grad():
            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                nesterov = group['nesterov']
                lr = group['lr']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    d_p = p.grad.data

                    if weight_decay != 0:
                        d_p = d_p.add(p.data, alpha=weight_decay)

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1)

                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    p.data.add_(d_p, alpha=-lr)

        return loss