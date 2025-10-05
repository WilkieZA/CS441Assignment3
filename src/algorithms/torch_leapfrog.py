import torch
from torch.optim.optimizer import Optimizer


class LeapFrogOptimizer(Optimizer):

    def __init__(self, params, dt=1e-2, delta_max=1e-1, xi=0.05, m=3,
                 mass=1.0, weight_decay=0.0):
        if dt <= 0.0:
            raise ValueError(f"Invalid dt: {dt}")
        if delta_max <= 0.0:
            raise ValueError(f"Invalid delta_max: {delta_max}")
        if xi <= 0.0:
            raise ValueError(f"Invalid xi: {xi}")
        if m < 1:
            raise ValueError(f"Invalid m: {m}")
        if mass <= 0.0:
            raise ValueError(f"Invalid mass: {mass}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(dt=dt, delta_max=delta_max, xi=xi, m=m,
                       mass=mass, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.state['succ_runs'] = 0
        self.state['nonpos_runs'] = 0
        self.state['first_step'] = True
        self.state['dt'] = dt

    def step(self, closure):
        loss = closure()
        loss.backward()

        params_list = []
        grads_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params_list.append(p)
                grads_list.append(p.grad.view(-1).detach().clone())

        x_k_flat = torch.cat([p.data.view(-1) for p in params_list])
        g_k_flat = torch.cat(grads_list)

        group = self.param_groups[0]
        delta_max = group['delta_max']
        xi = group['xi']
        m_thresh = group['m']
        mass = group['mass']
        weight_decay = group['weight_decay']
        dt = self.state['dt']

        if 'v_k' not in self.state:
            self.state['v_k'] = torch.zeros_like(x_k_flat)
            self.state['v_prev'] = torch.zeros_like(x_k_flat)
            self.state['x_prev'] = x_k_flat.clone()

        v_k = self.state['v_k']
        v_prev = self.state['v_prev']
        x_prev = self.state['x_prev']

        if weight_decay > 0:
            g_k_flat = g_k_flat + weight_decay * x_k_flat

        a_k = -g_k_flat / mass

        v_half = v_k + 0.5 * dt * a_k

        dx = dt * v_half
        step_norm = torch.norm(dx)

        cap_hit = False
        eff_scale = 1.0
        if step_norm > delta_max:
            eff_scale = delta_max / (step_norm + 1e-12)
            dx = dx * eff_scale
            cap_hit = True

        x_k1_flat = x_k_flat + dx
        self._set_params(params_list, x_k1_flat)

        new_loss = closure()
        new_loss.backward()

        g_k1_flat = torch.cat([p.grad.view(-1).detach().clone() for group in self.param_groups
                               for p in group['params'] if p.grad is not None])

        if weight_decay > 0:
            g_k1_flat = g_k1_flat + weight_decay * x_k1_flat

        a_k1 = -g_k1_flat / mass

        dt_eff = dt * eff_scale
        v_k1 = v_half + 0.5 * dt_eff * a_k1

        use_v_prev = v_prev if not self.state['first_step'] else v_k
        angle_metric = float(torch.dot(a_k1, use_v_prev))

        if angle_metric > 0.0 and not cap_hit:
            if not self.state['first_step']:
                self.state['succ_runs'] = self.state['succ_runs'] + 1
            else:
                self.state['succ_runs'] = 1

            p = 1.0 + self.state['succ_runs'] * xi
            self.state['dt'] = dt * p
            self.state['nonpos_runs'] = 0

        else:
            self.state['succ_runs'] = 1
            self.state['nonpos_runs'] += 1

            if self.state['nonpos_runs'] >= m_thresh:
                self.state['dt'] = dt * 0.5

                midpoint = 0.5 * (x_k_flat + x_prev)

                v_restart = 0.25 * (v_k1 + v_prev)

                self._set_params(params_list, midpoint)
                v_k1 = v_restart
                self.state['nonpos_runs'] = 0

                new_loss = loss

        self.state['x_prev'] = x_k_flat.clone()
        self.state['v_prev'] = v_k1.clone()
        self.state['v_k'] = v_k1.clone()
        self.state['first_step'] = False

        self.zero_grad()
        return new_loss

    def _set_params(self, params, flat):
        offset = 0
        for p in params:
            numel = p.numel()
            with torch.no_grad():
                p.data.copy_(flat[offset:offset + numel].view_as(p))
            offset += numel