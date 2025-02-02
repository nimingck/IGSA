import torch

from ..utils import *
from ..attack import Attack
import torch.nn.functional as F
from torchvision import transforms

class RAA(object):
    # with dta
    def __init__(self, model_name, targeted, n_theta=25, mu=0.04, K=10, u=1., decay=1., epoch=120, beta=1.5, lr=1.6/255):
        self.n_theta = n_theta  # Number of samples
        self.mu = mu   # Learning rate for sampling
        self.model = self.load_model(model_name).cuda()
        self.decay = decay
        self.eps_delta = 18/255
        self.eps_theta = 16/255
        self.labda = 0.1
        self.target = targeted
        self.step = epoch
        self.lr = lr
        self.K = K  # K of DTA
        self.u = u  # u of DTA
        self.epoch = epoch
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

    def forward(self, x, target):
        x = x.cuda()
        if(self.target):
            target = target[1].cuda()
        else:
            target = target.cuda()
        delta = torch.rand_like(x)
        delta = torch.clamp(delta, min=-self.eps_delta, max=self.eps_delta).cuda()
        
        for i in range(self.epoch):
            x_adv = x + delta
            self.model.zero_grad()
            sum_direction = torch.zeros_like(delta)
            
            for _ in range(self.n_theta):
                theta = self.labda * torch.randn_like(x_adv).cuda()
                theta.requires_grad = True
                loss_theta = F.cross_entropy(self.model(x_adv + theta), target, reduce=False, reduction="sum")
                loss_theta = torch.mean(loss_theta)
            
                loss_theta.backward(retain_graph=True)
                grad_theta = theta.grad.sign().detach()
                if(self.target):
                    thetanew = 0.05 * theta + self.mu * grad_theta
                else:
                    thetanew = 0.05 * theta - self.mu * grad_theta
                
                loss = F.cross_entropy(self.model(x_adv + thetanew), target, reduce=False, reduction="sum")
                loss_mean = torch.mean(loss)
                if(self.target):
                    direction_theta = loss.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, thetanew.size(1), thetanew.size(2), thetanew.size(3)) * thetanew
                else:      
                    direction_theta = -loss.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, thetanew.size(1), thetanew.size(2), thetanew.size(3)) * thetanew
                sum_direction += direction_theta.detach()
            
            # DTA's direction changes
            grad = sum_direction / self.n_theta
            gt = grad.clone().detach()
            delta_tk = delta.clone().detach()
            gtk = 0.
            momentum_tk = 0.
            delta_tk.requires_grad = True
            for k in range(self.K):
                # delta_tk.requires_grad = True
                grad = self.u * gt + grad / torch.norm(grad, p=1)
                gtk = gtk + grad
                momentum_tk = self.get_momentum(grad, momentum_tk)
                delta_tk = self.update_delta(delta_tk, x, momentum_tk, self.lr)
            
            grad = self.decay * gt + gtk / self.K
            delta = self.update_delta(delta, x, grad, self.lr)
            
            delta = torch.clamp(delta, min=-self.eps_delta, max=self.eps_delta)
            
            if i % 20 == 0:
                print("step: {}".format(i), "loss: {}".format((loss_mean / self.n_theta).detach()))
        
        delta = clamp(delta, self.img_min - x, self.img_max - x)
        return delta, self.model

    def update_delta(self, delta, data, momentum, alpha):
        return delta + alpha * momentum.sign()

    def get_momentum(self, grad, momentum):
        return self.u * momentum + grad
