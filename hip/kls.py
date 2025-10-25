import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

class KLSWithAuxAdam(torch.optim.Optimizer):
    """
    KLS

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.98)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['S'] below) moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner.
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer. 
            Helps at large precondition_frequency (~100 in our experiments), 
            but hurts performance at small precondition_frequency (~10 in our experiments).
        debias_ema (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias correction on running exponential-moving-average estimates of S, eigenvalues, and grad.
    """

    def __init__(
        self,
        param_groups,
    ):
        # lr: float = 3e-3, #tune this
        # betas=(0.9, 0.98), #tune this
        # weight_decay: float = 0.01, #tune this
        #####################
        eps: float = 1e-10, #optional (must set using_damping to True before tuning this)
        using_damping: bool = False, #optional
        precondition_frequency: int=10, #(it may perform better when using a smaller freq such as precondition_frequency=5)
        cast_dtype = torch.float32, #change this if you want to use bfloat16
        init_factor: float = 0.1, #optional
        #####################
        normalize_grads: bool = False, #do not change this
        debias_ema: bool = False, #do not change this
        using_shampoo_init: bool = False, #do not change this
        shampoo_beta: float= -1, #do not change this
        self.cast_dtype = cast_dtype
        self.using_shampoo_init = using_shampoo_init
        self.init_factor = init_factor
        print('init factor', self.init_factor)
        if self.using_shampoo_init:
            print('using shampoo init')
        else:
            print('using default init')

        self.using_damping = using_damping
        if using_damping:
            print('using damping in the curvature learning')
            self.damping = eps
        else:
            print('no damping in the curvature learning')
            self.damping = 0.0
        self.debias_ema = debias_ema
        
        for group in param_groups:
            assert "use_kls" in group
            if group["use_kls"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # tune these
                # group["lr"] = group["lr"]
                # group["betas"] = group["betas"]
                # group["weight_decay"] = group["weight_decay"]
                # fixed defaults
                group["shampoo_beta"] = shampoo_beta
                group["eps"] = eps
                group["precondition_frequency"] = precondition_frequency
                group["normalize_grads"] = normalize_grads
                assert set(group.keys()) == set(["params", "lr", "betas", "shampoo_beta", "eps", "weight_decay", "precondition_frequency", "normalize_grads", "use_kls"])
            else:
                # defaults for Adam
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_kls"])
        
        super().__init__(param_groups, dict())
        print('___________________________________KLS org v0___________________________________')


    def init_preconditioner(self, grad, state, precondition_frequency=10,
                            shampoo_beta=0.95):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        state['S'] = [] # Will hold all the preconditioner matrices (L and R in the paper).
        if grad.dim() == 1:
            state['S'].append([])
        else:
            for sh in grad.shape:
                state['S'].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))

        state['Q'] = None # Will hold all the eigenbases of the preconditioner.
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta

    def _get_ema_weights(self, state, beta_keep, param_name, step=True):
        """
        Returns the EMA weights to use for a stateful parameter.

        If `step` is True, updates the EMA bias tracker in the state.
        """
        beta_upd = 1.0 - beta_keep
        if not self.debias_ema:
            return beta_keep, beta_upd
        nm = f"shampoo_{param_name}_bias"
        if nm not in state:
            state[nm] = beta_upd ** state.get("step", 0)
        ema_bias = state[nm]
        ema_bias_new = ema_bias * beta_keep
        beta_keep = beta_keep * (1.0 - ema_bias) / (1.0 - ema_bias_new)
        beta_upd = beta_upd / (1.0 - ema_bias_new)
        if step:
            state[nm] = ema_bias_new
        return beta_keep, beta_upd


    @torch.compile
    def update_S(self, grad, state, mat, idx, beta_keep, beta_upd, total_factor, traces, total_trace, damping):
        factor = total_factor/grad.shape[idx]

        #update the curvature (every iteration)
        if damping > 0:
            trace = total_trace/traces[idx]
            state['S'][idx].mul_(beta_keep).add_(mat + (trace*factor)*torch.eye(state['S'][idx].shape[0], device=state['S'][idx].device), alpha=beta_upd/factor)
        else:
            state['S'][idx].mul_(beta_keep).add_(mat, alpha=beta_upd/factor)


    @torch.compile
    def update_eigen_value(self, state, diag, idx, beta_keep, beta_upd, traces, total_trace, damping):
        if damping > 0:
            diag = diag + total_trace/traces[idx]

        #update the eigen values of the curvature (every iteration)
        inv_d = state['eigen_sqrt_inv'][idx]**2
        D = torch.squeeze(1.0/inv_d).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        D.mul_(beta_keep).add_(diag, alpha=beta_upd)
        state['eigen_sqrt_inv'][idx] = (1.0/torch.sqrt(D)).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


    @torch.compile
    def update_4d_preconditioner(self, grad, state, total_factor, traces, total_trace,  damping):
        assert grad.dim() == 4 #only support 4d tensors
        invS_half = []
        for idx in range(grad.dim()):
            invS_h = state['Q'][idx] * state['eigen_sqrt_inv'][idx].view(1,-1)
            invS_half.append(invS_h)

        G = grad
        beta_keep_GG, beta_upd_GG = self._get_ema_weights(state, state["shampoo_beta"], "S")
        beta_keep_Ev, beta_upd_Ev = self._get_ema_weights(state, state["shampoo_beta"], "eigenvalues")

        GinvS1h = torch.einsum('ijab,ip->pjab', G, invS_half[1-1]) #G invS1
        GinvS12h = torch.einsum('pjab,jq->pqab', GinvS1h, invS_half[2-1])  #G invS1 invS2

        GinvS12Q3 = torch.einsum('pqab,al->pqlb', GinvS12h, state['Q'][3-1]) #G invS1 invS2 Q3
        GinvS12Q3Q4 = torch.einsum('pqlb,bm->pqlm', GinvS12Q3, state['Q'][4-1])#G invS1 invS2 Q3 Q4

        GinvS123h = GinvS12Q3 * state['eigen_sqrt_inv'][3-1].view(1,1,-1,1) #G invS1 invS2 invS3
        GinvS123G_T = torch.tensordot(GinvS123h,GinvS123h, dims=[[0,1,2],[0,1,2]]) #for S4
        self.update_S(G, state, GinvS123G_T, 4-1, beta_keep_GG, beta_upd_GG, total_factor, traces, total_trace, damping) #update S4

        GinvS12Q4 = torch.einsum('pqab,bm->pqam', GinvS12h, state['Q'][4-1]) #G invS1 invS2 Q4
        GinvS124h = GinvS12Q4 * state['eigen_sqrt_inv'][4-1].view(1,1,1,-1) #G invS1 invS2 invS4
        GinvS124G_T = torch.tensordot(GinvS124h,GinvS124h, dims=[[0,1,3],[0,1,3]]) #for S3
        self.update_S(G, state, GinvS124G_T, 3-1, beta_keep_GG, beta_upd_GG, total_factor, traces, total_trace, damping) #update S3

        diag4 = torch.mean( (GinvS12Q3Q4*state['eigen_sqrt_inv'][3-1].view(1,1,-1,1))**2, dim=(0,1,2) ) #torch.diag( Q4.T @ GinvS123G_T @ Q4 )/(d1*d2*d3) #for diag(S4)
        self.update_eigen_value(state, diag4, 4-1, beta_keep_Ev, beta_upd_Ev, traces, total_trace, damping)

        diag3 = torch.mean( (GinvS12Q3Q4*state['eigen_sqrt_inv'][4-1].view(1,1,1,-1))**2, dim=(0,1,3) ) #torch.diag( Q3.T @ GinvS124G_T @ Q3 )/(d1*d2*d4) #for diag(S3)
        self.update_eigen_value(state, diag3, 3-1, beta_keep_Ev, beta_upd_Ev, traces, total_trace, damping)

        GinvS4h = torch.einsum('ijab,bm->ijam', G, invS_half[4-1]) #G invS4
        GinvS43h = torch.einsum('ijam,al->ijlm', GinvS4h, invS_half[3-1]) #G invS4 invS3

        GinvS43Q2 = torch.einsum('ijlm,jq->iqlm', GinvS43h, state['Q'][2-1])
        GinvS43Q2Q1 = torch.einsum('iqlm,ip->pqlm', GinvS43Q2, state['Q'][1-1]) #G invS4 invS3 Q2 Q1

        GinvS432h = GinvS43Q2 * state['eigen_sqrt_inv'][2-1].view(1,-1,1,1) #G invS4 invS3 invS2
        GinvS432G_T = torch.tensordot(GinvS432h,GinvS432h, dims=[[1,2,3],[1,2,3]]) #for S1
        self.update_S(G, state, GinvS432G_T, 1-1, beta_keep_GG, beta_upd_GG, total_factor, traces, total_trace, damping) #update S1

        GinvS43Q1 = torch.einsum('ijlm,ip->pjlm', GinvS43h, state['Q'][1-1])
        GinvS431h = GinvS43Q1 * state['eigen_sqrt_inv'][1-1].view(-1,1,1,1) #G invS4 invS3 invS1
        GinvS431G_T = torch.tensordot(GinvS431h,GinvS431h, dims=[[0,2,3],[0,2,3]]) #for S2
        self.update_S(G, state, GinvS431G_T, 2-1, beta_keep_GG, beta_upd_GG, total_factor, traces, total_trace, damping) #update S2

        diag1 = torch.mean( (GinvS43Q2Q1*state['eigen_sqrt_inv'][2-1].view(1,-1,1,1))**2, dim=(1,2,3) ) #torch.diag( Q1.T @ GinvS432G_T @ Q1 )/(d2*d3*d4) #for diag(S1)
        self.update_eigen_value(state, diag1, 1-1, beta_keep_Ev, beta_upd_Ev, traces, total_trace, damping)

        diag2 = torch.mean( (GinvS43Q2Q1*state['eigen_sqrt_inv'][1-1].view(-1,1,1,1))**2, dim=(0,2,3) ) #torch.diag( Q2.T @ GinvS431G_T @ Q2 )/(d1*d3*d4) #for diag(S2)
        self.update_eigen_value(state, diag2, 2-1, beta_keep_Ev, beta_upd_Ev, traces, total_trace, damping)


    @torch.compile
    def update_3d_preconditioner(self, grad, state, total_factor, traces, total_trace,  damping):
        assert grad.dim() == 3 #only support 3d tensors
        invS_half = []
        for idx in range(grad.dim()):
            invS_h = state['Q'][idx] * state['eigen_sqrt_inv'][idx].view(1,-1)
            invS_half.append(invS_h)

        G = grad
        beta_keep_GG, beta_upd_GG = self._get_ema_weights(state, state["shampoo_beta"], "S")
        beta_keep_Ev, beta_upd_Ev = self._get_ema_weights(state, state["shampoo_beta"], "eigenvalues")

        GinvS1h = torch.einsum('ija,ip->pja', G, invS_half[1-1]) #G invS1
        GinvS1Q2 = torch.einsum('pja,jl->pla', GinvS1h, state['Q'][2-1]) #G invS1 Q2
        GinvS1Q2Q3 = torch.einsum('pqa,am->pqm', GinvS1Q2, state['Q'][3-1])#G invS1 Q2 Q3

        GinvS12h = GinvS1Q2 * state['eigen_sqrt_inv'][2-1].view(1,-1,1) #G invS1 invS2
        GinvS12G_T = torch.tensordot(GinvS12h,GinvS12h, dims=[[0,1],[0,1]]) #for S3
        self.update_S(G, state, GinvS12G_T, 3-1, beta_keep_GG, beta_upd_GG, total_factor, traces, total_trace, damping) #update S3

        GinvS1Q3 = torch.einsum('pqb,bm->pqm', GinvS1h, state['Q'][3-1]) #G invS1 Q3
        GinvS13h = GinvS1Q3 * state['eigen_sqrt_inv'][3-1].view(1,1,-1) #G invS1 invS3
        GinvS13G_T = torch.tensordot(GinvS13h,GinvS13h, dims=[[0,2],[0,2]]) #for S2
        self.update_S(G, state, GinvS13G_T, 2-1, beta_keep_GG, beta_upd_GG, total_factor, traces, total_trace, damping) #update S2

        diag3 = torch.mean( (GinvS1Q2Q3*state['eigen_sqrt_inv'][2-1].view(1,-1,1))**2, dim=(0,1) ) #torch.diag( Q3.T @ GinvS12G_T @ Q3 )/(d1*d2) #for diag(S3)
        self.update_eigen_value(state, diag3, 3-1, beta_keep_Ev, beta_upd_Ev, traces, total_trace, damping)

        diag2 = torch.mean( (GinvS1Q2Q3*state['eigen_sqrt_inv'][3-1].view(1,1,-1))**2, dim=(0,2) ) #torch.diag( Q2.T @ GinvS13G_T @ Q2 )/(d1*d3) #for diag(S2)
        self.update_eigen_value(state, diag2, 2-1, beta_keep_Ev, beta_upd_Ev, traces, total_trace, damping)


        GinvS3h = torch.einsum('ijb,bm->ijm', G, invS_half[3-1]) #G invS3
        GinvS3Q2 = torch.einsum('ijm,jq->iqm', GinvS3h, state['Q'][2-1])
        GinvS3Q2Q1 = torch.einsum('iqm,ip->pqm', GinvS3Q2, state['Q'][1-1]) #G invS3 Q2 Q1

        GinvS32h = GinvS3Q2 * state['eigen_sqrt_inv'][2-1].view(1,-1,1) #G invS3 invS2
        GinvS32G_T = torch.tensordot(GinvS32h,GinvS32h, dims=[[1,2],[1,2]]) #for S1
        self.update_S(G, state, GinvS32G_T, 1-1, beta_keep_GG, beta_upd_GG, total_factor, traces, total_trace, damping) #update S1

        diag1 = torch.mean( (GinvS3Q2Q1*state['eigen_sqrt_inv'][2-1].view(1,-1,1))**2, dim=(1,2) ) #torch.diag( Q1.T @ GinvS32G_T @ Q1 )/(d2*d3) #for diag(S1)
        self.update_eigen_value(state, diag1, 1-1, beta_keep_Ev, beta_upd_Ev, traces, total_trace, damping)


    @torch.compile
    def update_2d_preconditioner(self, grad, state, total_factor, traces, total_trace, damping):
        assert grad.dim() == 2 #only support 2d tensors

        beta_keep_GG, beta_upd_GG = self._get_ema_weights(state, state["shampoo_beta"], "S")
        for idx, sh in enumerate(grad.shape):
            o = state['Q'][abs(idx-1)]
            sqrt_inv_d = state['eigen_sqrt_inv'][abs(idx-1)]

            if idx == 0: #(left)
                step0 = o.T @ grad.T
                lhalf = step0 * sqrt_inv_d.view(-1,1)
                mat = lhalf.T @ lhalf
            else: #(right)
                step1 = o.T @ grad
                rhalf= step1 * sqrt_inv_d.view(-1,1)
                mat = rhalf.T @ rhalf

            self.update_S(grad, state, mat, idx, beta_keep_GG, beta_upd_GG, total_factor, traces, total_trace, damping)

        beta_keep_Ev, beta_upd_Ev = self._get_ema_weights(state, state["shampoo_beta"], "eigenvalues")
        # diag_half = state['Q'][0].T @ grad @ state['Q'][1]
        diag_half = step1 @ state['Q'][1]
        ldiag = torch.mean( (diag_half * state['eigen_sqrt_inv'][1].view(1,-1))**2, 1)
        rdiag = torch.mean( (diag_half * state['eigen_sqrt_inv'][0].view(-1,1))**2, 0)

        self.update_eigen_value(state, ldiag, 1-1, beta_keep_Ev, beta_upd_Ev, traces, total_trace, damping)
        self.update_eigen_value(state, rdiag, 2-1, beta_keep_Ev, beta_upd_Ev, traces, total_trace, damping)


    @torch.no_grad()
    def update_preconditioner(self, grad, state):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        if self.using_damping:
            damping = self.damping
        else:
            damping = 0.0

        traces = []
        total_factor = torch.numel(grad)
        total_trace = damping
        if damping > 0:
            for idx, sh in enumerate(grad.shape):
                if state['Q'] is None:
                    cur_trace = 1.0 #average
                else:
                    cur_trace = torch.mean(state['eigen_sqrt_inv'][idx]**2) #average
                total_trace *= cur_trace
                traces.append(cur_trace)

        if state['Q'] is None:
            beta_keep_GG, beta_upd_GG = self._get_ema_weights(state, state["shampoo_beta"], "S")
            tmp = []
            for idx, sh in enumerate(grad.shape):
                mat = torch.tensordot(
                        grad,
                        grad,
                        # Contracts across all dimensions except for k.
                        dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                    )
                tmp.append(mat)
                if self.using_shampoo_init:
                    # beta2 = 1.0 - state['shampoo_beta']
                    #state['S'][idx].mul_(1-beta2).add_(mat, alpha=beta2)
                    state['S'][idx].mul_(beta_keep_GG).add_(mat, alpha=beta_upd_GG)
                else:
                    self.update_S(grad, state, mat, idx, beta_keep_GG, beta_upd_GG, total_factor, traces, total_trace, damping)

            state['Q'], state['eigen_sqrt_inv'] = self.get_orthogonal_matrix(tmp)
            del tmp
        else:
            if len(grad.shape)==2:
                self.update_2d_preconditioner(grad, state, total_factor, traces, total_trace,  damping)
            elif len(grad.shape)==3:
                self.update_3d_preconditioner(grad, state, total_factor, traces, total_trace,  damping)
            elif len(grad.shape)==4:
                self.update_4d_preconditioner(grad, state, total_factor, traces, total_trace,  damping)
            else:
                assert False

        if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            #update the eigen bases of the curvature (every k iterations)
            state['Q'] = self.get_orthogonal_matrix_QR(state)


    @torch.no_grad()
    def step(self, closure = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()
        
        for group in self.param_groups:
            if group["use_kls"]:
                is_merged = False
                for p in group["params"]:
                    if 'merged' in group and group['merged']:
                        is_merged = True

                    if p.grad is None:
                        continue

                    if is_merged:
                        total_idx = len(group['params'])
                        grad = torch.stack([ k.grad.to(dtype=self.cast_dtype) for k in group['params'] ] )
                        assert grad.shape[0] == total_idx
                    else:
                        grad = torch.squeeze( p.grad.to(dtype=self.cast_dtype) )

                    assert len(grad.shape) >= 2
                    state = self.state[p]
                    
                    if "step" not in state:
                        state["step"] = 0
                        if is_merged: print('using 3d merged-----------------', total_idx, grad.shape)
                        
                    # State initialization
                    if "exp_avg" not in state:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)

                    if 'Q' not in state:
                        self.init_preconditioner(
                            grad,
                            state,
                            precondition_frequency=group['precondition_frequency'],
                            shampoo_beta=(group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group["betas"][1]),
                        )
                        self.update_preconditioner(grad, state)
                        continue # first step is skipped so that we never use the current gradients in the projection.
                    
                    exp_avg = state["exp_avg"]
                    state["step"] += 1
                    beta_keep_grad, beta_upd_grad = self._get_ema_weights(state, group["betas"][0], "grad")
                    exp_avg.mul_(beta_keep_grad).add_(grad, alpha=beta_upd_grad)

                    grad_projected = self.project(exp_avg, state)
                    if len(grad.shape) == 2:
                        precond_grad = (grad_projected * state['eigen_sqrt_inv'][0].view(-1,1)) * state['eigen_sqrt_inv'][1].view(1,-1)
                    elif len(grad.shape) == 3:
                        precond_grad = (grad_projected * state['eigen_sqrt_inv'][0].view(-1,1,1)) * state['eigen_sqrt_inv'][1].view(1,-1,1) * state['eigen_sqrt_inv'][2].view(1,1,-1)
                    elif len(grad.shape) == 4:
                        precond_grad = (grad_projected * state['eigen_sqrt_inv'][0].view(-1,1,1,1)) * state['eigen_sqrt_inv'][1].view(1,-1,1,1) * state['eigen_sqrt_inv'][2].view(1,1,-1,1) * state['eigen_sqrt_inv'][3].view(1,1,1,-1)
                    else:
                        assert False
                    norm_grad = self.project_back(precond_grad, state)

                    if group["normalize_grads"]:
                        norm_grad = norm_grad / (1e-30+torch.mean(norm_grad**2)**0.5)
                    
                    step_size = group["lr"]

                    if is_merged:
                        for idx in range(total_idx):
                            group['params'][idx].add_(norm_grad[idx,:,:].view(group['params'][idx].shape), alpha=-step_size)
                    else:
                        p.add_(norm_grad.view(p.shape), alpha=-step_size)
                    
                    if group["weight_decay"] > 0.0:
                        if is_merged:
                            for idx in range(total_idx):
                                group['params'][idx].add_(group['params'][idx], alpha=(-group["lr"] * group["weight_decay"]))
                        else:
                            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                        
                    # Update is done after the gradient step to avoid using current gradients in the projection.
                    self.update_preconditioner(grad, state)

                    if is_merged:
                        break
            else:
                # Adam
                # taken from MuonWithAuxAdam
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss


    def project(self, grad, state):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        original_shape = grad.shape

        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [0]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        
        return grad
 
    def project_back(self, grad, state):
        """
        Projects the gradient back to the original space.
        """
        original_shape = grad.shape
        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [1]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
                
        return grad
        

    def get_orthogonal_matrix(self, mat, update_GG=False):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        """
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)
        
        final = []
        info = []
        assert len(matrix) == len(mat)
        for idx, m in enumerate(matrix):
            if len(m) == 0:
                final.append([])
                continue
            try:
                v0, Q = torch.linalg.eigh(m+1e-30*torch.eye(m.shape[0], device=m.device))
            except:
                v0, Q = torch.linalg.eigh(m.to(torch.float64)+1e-30*torch.eye(m.shape[0], device=m.device))
                Q = Q.to(m.dtype)
                v0 = v0.to(m.dtype)
            if update_GG:
                v0[ v0 < 0 ] = 0.0
                mat[idx] = (Q @ torch.diag( torch.sqrt(v0) ) @ Q.T).to(original_type)

            Q = torch.flip(Q, [1])
            v = torch.ones(Q.shape[0], device=Q.device, dtype=Q.dtype)*self.init_factor

            sqrt_inv_d = (1.0/torch.sqrt(v)).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if not float_data:
                sqrt_inv_d = sqrt_inv_d.to(device=original_device, dtype=original_type)

            info.append(sqrt_inv_d)

            if not float_data:
                Q = Q.to(device=original_device, dtype=original_type)
            final.append(Q)
        return final, info
        

    def get_orthogonal_matrix_QR(self, state):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration
        followed by torch.linalg.qr decomposition.
        """
        precond_list = state['S']
        orth_list = state['Q']

        matrix = []
        orth_matrix = []
        for m,o in zip(precond_list, orth_list):
            assert len(m) > 0
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
        
        final = []
        for ind, (m,o) in enumerate(zip(matrix, orth_matrix)):
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)

            if not float_data:
                Q = Q.to(device=original_device, dtype=original_type)
            final.append(Q)
        return final

