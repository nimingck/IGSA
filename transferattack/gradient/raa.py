import torch
from ..utils import *
from ..attack import Attack
import torch.nn.functional as F
from torchvision import transforms


class RAA(object):
    def __init__(self, model_name, targeted, n_theta=20, mu=0.4):
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
        self.eps_delta = 16 / 255  # Perturbation budget for delta
        self.eps_theta = 16 / 255  # Perturbation budget for theta
        self.labda = 0.1  # Scaling factor for theta
        self.target = targeted  # Whether the attack is targeted
        self.step = 200  # Number of optimization steps
        self.lr = 1.6 / 255  # Learning rate for updating delta
        self.img_max, self.img_min = 1., 0  # Bounds for image pixel values

    def load_model(self, model_name):
        """
        Load the surrogate model. This method should be overridden if a custom surrogate model is used (e.g., DSM, SETR, etc.).
        It prioritizes models from torchvision.models, then timm.models.

        Arguments:
            model_name (str/list): The name of the surrogate model (or a list of model names) from model_list in utils.py.

        Returns:
            model (torch.nn.Module): The surrogate model wrapped by wrap_model in utils.py.
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
        """
        Perform the adversarial attack to generate perturbed inputs.

        Arguments:
            x (torch.Tensor): The input images to perturb.
            target (torch.Tensor): The ground-truth labels (or target labels for targeted attacks).

        Returns:
            delta (torch.Tensor): The generated adversarial perturbation.
            model: The surrogate model used for the attack.
        """
        x = x.cuda()
        if self.target:
            target = target[1].cuda()
        else:
            target = target.cuda()

        # 1. Initialize delta and constrain it
        delta = torch.rand_like(x)
        delta = torch.clamp(delta, min=-self.eps_delta, max=self.eps_delta).cuda()

        for i in range(self.step):
            x_adv = x + delta
            loss_print = 0
            self.model.zero_grad()
            sum_direction = torch.zeros_like(delta)

            for _ in range(self.n_theta):
                # 2. Sample n different thetas from a Gaussian distribution and constrain them
                theta = self.labda * torch.randn_like(x_adv).cuda()
                # theta = torch.clamp(theta, min=-self.eps_theta, max=self.eps_theta)

                # 3. Compute L_theta for each theta
                theta.requires_grad = True
                loss_theta = F.cross_entropy(self.model(x_adv + theta), target, reduce=False, reduction="sum")
                loss_theta = torch.mean(loss_theta)

                # 4. Update thetanew and constrain it
                loss_theta.backward(retain_graph=True)
                grad_theta = theta.grad.sign().detach()
                if self.target:
                    thetanew = 0.05 * theta + self.mu * grad_theta
                else:
                    thetanew = 0.05 * theta - self.mu * grad_theta
                
                # 5. Compute direction_theta = L(x + delta + thetanew) * theta
                loss = F.cross_entropy(self.model(x_adv + thetanew), target, reduce=False, reduction="sum")
                loss_mean = torch.mean(loss)
                if self.target:
                    direction_theta = loss.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, thetanew.size(1), thetanew.size(2), thetanew.size(3)) * thetanew
                else:
                    direction_theta = -loss.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, thetanew.size(1), thetanew.size(2), thetanew.size(3)) * thetanew
                # 6. Accumulate the direction
                sum_direction += direction_theta.detach()

                loss_print += loss_theta

            # 7. Update delta
            if self.target:
                delta = delta - (sum_direction / self.n_theta).sign() * self.lr
            else:
                delta = delta + (sum_direction / self.n_theta).sign() * self.lr
            # delta = delta - (sum_direction / self.n_theta)*0.01

            # 8. Constrain delta
            delta = torch.clamp(delta, min=-self.eps_delta, max=self.eps_delta)

            if i % 20 == 0:
                print("step: {}".format(i), "loss: {}".format((loss_print / self.n_theta).detach()))

        delta = clamp(delta, self.img_min - x, self.img_max - x)

        return delta, self.model