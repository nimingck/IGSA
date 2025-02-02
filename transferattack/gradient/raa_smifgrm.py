import torch

from ..utils import *
from ..attack import Attack
import torch.nn.functional as F
from torchvision import transforms

class RAA(object):
    # with smifgrm
    def __init__(self, model_name, targeted, n_theta=25, mu=0.04, num_neighbor=12, rescale_factor=2, epoch=100, lr=1.6/255):
        """
        Initialize the RAA (Randomized Adversarial Attack) class.

        Arguments:
            model_name (str): The name of the surrogate model to use.
            targeted (bool): Whether the attack is targeted (True) or untargeted (False).
            n_theta (int, optional): The number of samples for randomization. Defaults to 20.
            mu (float, optional): The learning rate for sampling. Defaults to 0.4.
        """
        self.n_theta = n_theta  # Number of samples
        self.mu = mu  # Learning rate for sampling
        self.model = self.load_model(model_name).cuda()
        self.eps_delta = 16/255
        self.eps_theta = 16/255
        self.labda = 0.1
        self.target = targeted
        self.step = epoch
        self.lr = lr
        self.num_neighbor = num_neighbor  # Sampling nums of SMI-FGRM
        self.rescale_factor = rescale_factor  # Rescale_factor of SMI-FGRM
        self.img_max, self.img_min = 1., 0

    
    def load_model(self, model_name):
        """
        The model Loading stage, which should be overridden when surrogate model is customized (e.g., DSM, SETR, etc.)
        Prioritize the model in torchvision.models, then timm.models

        Arguments:
            model_name (str/list): the name of surrogate model in model_list in utils.py

        Returns:
            model (torch.nn.Module): the surrogate model wrapped by wrap_model in utils.py
        """
        def load_single_model(model_name):
            if model_name in models.__dict__.keys():
                print('=> Loading model {} from torchvision.models'.format(model_name))
                model = models.__dict__[model_name](weights="DEFAULT")
            elif model_name in timm.list_models():
                print('=> Loading model {} from timm.models'.format(model_name))
                model = timm.create_model(model_name, pretrained=True)
            else:
                raise ValueError('Model {} not supported'.format(model_name))
            return wrap_model(model.eval().cuda())

        if isinstance(model_name, list):
            return EnsembleModel([load_single_model(name) for name in model_name])
        else:
            return load_single_model(model_name)

    def rescale_grad(self, grad):
        log_abs_grad = grad.abs().log2()
        grad_mean = torch.mean(log_abs_grad, dim=(1,2,3), keepdim=True)
        grad_std = torch.std(log_abs_grad, dim=(1,2,3), keepdim=True)
        norm_grad = (log_abs_grad - grad_mean) / grad_std
        return self.rescale_factor * grad.sign() * torch.sigmoid(norm_grad)

    def forward(self, x, target):
        x = x.cuda()
        if(self.target):
            target = target[1].cuda()
        else:
            target = target.cuda()
        delta = torch.rand_like(x)
        delta = torch.clamp(delta, min=-self.eps_delta, max=self.eps_delta).cuda()
        
        for i in range(self.step):
            x_adv = x + delta
            self.model.zero_grad()
            sum_direction = torch.zeros_like(delta)
            
            for _ in range(self.n_theta):
                theta = self.labda * torch.randn_like(x_adv).cuda()
                theta.requires_grad = True
                loss_theta = F.cross_entropy(self.model(x_adv + theta), target, reduce=False, reduction='sum')
                loss_theta = torch.mean(loss_theta)

                loss_theta.backward(retain_graph=True)
                grad_theta = theta.grad.sign().detach()
                if(self.target):
                    thetanew = 0.05 * theta + self.mu * grad_theta
                else:
                    thetanew = 0.05 * theta - self.mu * grad_theta
                
                loss = F.cross_entropy(self.model(x_adv + thetanew), target, reduce=False, reduction='sum')
                loss_mean = torch.mean(loss)
                if(self.target):
                    direction_theta = loss.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, thetanew.size(1), thetanew.size(2), thetanew.size(3)) * thetanew
                else:      
                    direction_theta = -loss.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, thetanew.size(1), thetanew.size(2), thetanew.size(3)) * thetanew
                sum_direction += direction_theta.detach()
            
            grad = sum_direction / self.n_theta
            grad = self.rescale_grad(grad)
            if(self.target):
                delta = delta - self.lr * grad
            else:
                delta = delta + self.lr * grad
            delta = torch.clamp(delta, min=-self.eps_delta, max=self.eps_delta)
            
            if i % 20 == 0:
                print("step: {}".format(i), "loss: {}".format(loss_mean.item()))

        delta = torch.clamp(delta, min=self.img_min-x, max=self.img_max-x)
        return delta, self.model