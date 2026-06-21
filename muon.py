import math
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Muon core utilities.

def _is_single_rank_dtensor(tensor):
    if not all(hasattr(tensor, attr) for attr in ("to_local", "device_mesh", "placements")):
        return False
    try:
        return tensor.device_mesh.size() == 1
    except Exception:
        return False

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Uses the standard quintic Muon coefficients.
    """
    assert G.ndim >= 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    use_local_dtensor = _is_single_rank_dtensor(G)
    if use_local_dtensor:
        from torch.distributed._tensor import DTensor

        device_mesh = G.device_mesh
        placements = G.placements
        global_shape = G.shape
        G_work = G.to_local()
    else:
        DTensor = None
        device_mesh = None
        placements = None
        global_shape = None
        G_work = G

    out_dtype = G_work.dtype
    X = G_work.float()

    if G_work.size(-2) > G_work.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for idx in range(1, steps + 1):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G_work.size(-2) > G_work.size(-1):
        X = X.mT

    X = X.to(dtype=out_dtype)
    if use_local_dtensor:
        X = DTensor.from_local(
            X,
            device_mesh=device_mesh,
            placements=placements,
            shape=global_shape,
            stride=X.stride(),
        )
    return X

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = torch.lerp(grad, momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update

def _muon_update_param(p, state, group):
    if p.grad is None:
        p.grad = torch.zeros_like(p)
    if len(state) == 0:
        state["momentum_buffer"] = torch.zeros_like(p)

    beta = group["momentum"]
    momentum = state["momentum_buffer"]
    momentum.lerp_(p.grad, 1 - beta)

    update = torch.lerp(p.grad, momentum, beta) if group.get("nesterov", True) else momentum

    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=group.get("ns_steps", 5))

    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

    p.mul_(1 - group["lr"] * group["weight_decay"])
    p.add_(update.reshape(p.shape), alpha=-group["lr"])

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    _muon_update_param(p, self.state[p], group)
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss

class SingleDeviceMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                _muon_update_param(p, self.state[p], group)

        return loss

def _make_aux_adam(param_groups):
    aux_groups = []
    for group in param_groups:
        if not group["use_muon"]:
            aux_groups.append({
                "params": group["params"],
                "lr": group.get("lr", 1e-3),
                "betas": group.get("betas", (0.9, 0.999)),
                "eps": group.get("eps", 1e-8),
                "weight_decay": group.get("weight_decay", 0),
            })
    return None if len(aux_groups) == 0 else torch.optim.AdamW(aux_groups)

class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        _muon_update_param(p, self.state[p], group)
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss

class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    _muon_update_param(p, self.state[p], group)

        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss

class MuonWithAuxSGD(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        _muon_update_param(p, self.state[p], group)
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss

class SingleDeviceMuonWithAuxSGD(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "nesterov", "ns_steps", "use_muon"])
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        torch.optim.Optimizer.__init__(self, param_groups, dict())
        self.aux_adam = _make_aux_adam(self.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    _muon_update_param(p, self.state[p], group)
        if self.aux_adam is not None:
            self.aux_adam.step()

        return loss
