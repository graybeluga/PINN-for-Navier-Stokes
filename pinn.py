import torch
from torch import nn, autograd, Tensor
from torch.nn import functional as F


def gradients(y, x) -> Tensor:
    grad = autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True,)[0]
    return grad


class FFNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inter_dim = 4 * dim
        self.fc1 = nn.Linear(dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, dim)
        self.act_fn = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x0 = x
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + x0


class Pinn(nn.Module):
    def __init__(self, min_x: int, max_x: int):
        super().__init__()

        self.MIN_X = min_x
        self.MAX_X = max_x

        # Build FFN network
        self.hidden_dim = 128
        self.num_blocks = 8
        self.first_map = nn.Linear(3, self.hidden_dim)
        self.last_map = nn.Linear(self.hidden_dim, 2)
        self.ffn_blocks = nn.ModuleList([
            FFNN(self.hidden_dim) for _ in range(self.num_blocks)
        ])

        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(0.01))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def ffn(self, inputs: Tensor) -> Tensor:
        x = self.first_map(inputs)
        for blk in self.ffn_blocks:
            x = blk(x)
        x = self.last_map(x)
        return x

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        p: Tensor = None,
        u: Tensor = None,
        v: Tensor = None,
    ):
        """
        All shapes are (b,)
        inputs: x, y, t
        labels: p, u, v
        """
        inputs = torch.stack([x, y, t], dim=1)
        inputs = 2.0 * (inputs - self.MIN_X) / (self.MAX_X - self.MIN_X) - 1.0

        hidden_output = self.ffn(inputs)
        psi = hidden_output[:, 0]
        p_hat = hidden_output[:, 1]
        u_hat = gradients(psi, y)
        v_hat = -gradients(psi, x)

        preds = torch.stack([p_hat, u_hat, v_hat], dim=1)
        u_t = gradients(u_hat, t)
        u_x = gradients(u_hat, x)
        u_y = gradients(u_hat, y)
        u_xx = gradients(u_x, x)
        u_yy = gradients(u_y, y)

        v_t = gradients(v_hat, t)
        v_x = gradients(v_hat, x)
        v_y = gradients(v_hat, y)
        v_xx = gradients(v_x, x)
        v_yy = gradients(v_y, y)

        p_x = gradients(p_hat, x)
        p_y = gradients(p_hat, y)

        f_u = (u_t + self.lambda1 * (u_hat * u_x + v_hat * u_y) + p_x - self.lambda2 * (u_xx + u_yy))
        f_v = (v_t + self.lambda1 * (u_hat * v_x + v_hat * v_y) + p_y - self.lambda2 * (v_xx + v_yy))

        loss, losses = self.loss_fn(u, v, u_hat, v_hat, f_u, f_v)
        return {
            "preds": preds,
            "loss": loss,
            "losses": losses,
        }

    def loss_fn(self, u, v, u_hat, v_hat, f_u_hat, f_v_hat):
        """
        u: (b, 1)
        v: (b, 1)
        p: (b, 1)
        """
        u_loss = F.mse_loss(u_hat, u)
        v_loss = F.mse_loss(v_hat, v)
        f_u_loss = F.mse_loss(f_u_hat, torch.zeros_like(f_u_hat))
        f_v_loss = F.mse_loss(f_v_hat, torch.zeros_like(f_v_hat))
        loss = u_loss + v_loss + f_u_loss + f_v_loss
        return loss, {
            "u_loss": u_loss,
            "v_loss": v_loss,
            "f_u_loss": f_u_loss,
            "f_v_loss": f_v_loss,
        }