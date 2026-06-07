import math
import torch
from torch.optim.optimizer import Optimizer


class MergedAdaBelief(Optimizer):
    """
    Merged AdaBelief Optimizer
    = Innovation 1 + Innovation 2

    创新点1：
        使用历史动量 m_{t-1} 作为残差参考：
            r_t = g_t - m_{t-1}

    创新点2：
        使用部分自适应分母：
            denom = (s_hat) ^ p_adapt + eps
        其中 p_adapt ∈ [0, 0.5]

    当：
        1) r_t = g_t - m_t
        2) p_adapt = 0.5
    时，可退化回标准 AdaBelief 形式。

    参数说明：
        params            : 待优化参数
        lr                : 学习率
        betas             : (beta1, beta2)
        eps               : 数值稳定项
        weight_decay      : 权重衰减系数
        weight_decouple   : 是否使用解耦权重衰减
        p_adapt           : 部分自适应指数，范围 [0, 0.5]
        amsgrad           : 是否启用 AMSGrad
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        weight_decouple=True,
        p_adapt=0.497,
        amsgrad=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= p_adapt <= 0.5:
            raise ValueError(f"Invalid p_adapt: {p_adapt}, should be in [0, 0.5]")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            p_adapt=p_adapt,
            amsgrad=amsgrad,
        )
        super(MergedAdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MergedAdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("weight_decouple", True)
            group.setdefault("p_adapt", 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]
            p_adapt = group["p_adapt"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MergedAdaBelief does not support sparse gradients")

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]              # m_t
                exp_avg_var = state["exp_avg_var"]      # s_t

                state["step"] += 1
                step = state["step"]

                # -------------------------------------------------
                # 1) 权重衰减
                # -------------------------------------------------
                if weight_decay != 0:
                    if weight_decouple:
                        # AdamW风格：解耦权重衰减
                        p.mul_(1.0 - lr * weight_decay)
                    else:
                        # 耦合式L2正则
                        grad = grad.add(p, alpha=weight_decay)

                # -------------------------------------------------
                # 2) 保存旧动量 m_{t-1}
                # -------------------------------------------------
                m_prev = exp_avg.clone()

                # -------------------------------------------------
                # 3) 更新一阶动量 m_t
                #    m_t = beta1 * m_{t-1} + (1-beta1) * g_t
                # -------------------------------------------------
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # -------------------------------------------------
                # 4) 创新点1：历史动量参考残差
                #    r_t = g_t - m_{t-1}
                # -------------------------------------------------
                grad_residual = grad - m_prev

                # -------------------------------------------------
                # 5) 更新 belief 二阶项
                #    s_t = beta2 * s_{t-1} + (1-beta2) * r_t^2
                # -------------------------------------------------
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1.0 - beta2
                )

                # -------------------------------------------------
                # 6) 偏置修正
                # -------------------------------------------------
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                m_hat = exp_avg / bias_correction1

                if amsgrad:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    torch.maximum(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    s_hat = max_exp_avg_var / bias_correction2
                else:
                    s_hat = exp_avg_var / bias_correction2

                # -------------------------------------------------
                # 7) 创新点2：部分自适应分母
                #    denom = (s_hat)^p_adapt + eps
                #    当 p_adapt = 0.5 时，约等于标准 AdaBelief 的 sqrt(s_hat)
                # -------------------------------------------------
                denom = s_hat.pow(p_adapt).clamp_min(eps)

                # -------------------------------------------------
                # 8) 参数更新
                #    theta_t = theta_{t-1} - lr * m_hat / denom
                # -------------------------------------------------
                p.addcdiv_(m_hat, denom.add(eps), value=-lr)

        return loss