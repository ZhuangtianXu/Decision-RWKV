import torch, types, os, gc, math
from torch.cuda.amp import autocast
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_utils import PreTrainedModel  #, Conv1D
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

args = types.SimpleNamespace()

# args.n_layer = 4
# # args.n_embd = 2048
# args.n_embd = 512

# args.dim_att = args.n_embd
# args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

# args.vocab_size = 65536


args.ctx_len = 4096
# args.ctx_len = 2000

args.head_size_a = 64 # don't change
args.head_size_divisor = 8 # don't change


from torch.utils.cpp_extension import load


wkv6_cuda = load(name="wkv6", sources=[
        "/root/autodl-tmp/DR/gym/cuda/wkv6_op.cpp",
        "/root/autodl-tmp/DR/gym/cuda/wkv6_cuda.cu",
        # "/root/autodl-tmp/DR/gym/cuda/wkv6state_op.cpp",
        # "/root/autodl-tmp/DR/gym/cuda/wkv6state_cuda.cu"
    ],
    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={args.head_size_a}", f"-D_T_={args.ctx_len}"])



class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u): # forward: r, k, v, w, u => y
        with torch.no_grad():
            r = r.contiguous().bfloat16()
            # r = r.to(torch.bfloat16)
            assert r.dtype == torch.bfloat16
            k = k.contiguous().bfloat16()
            # k = k.to(torch.bfloat16)
            assert k.dtype == torch.bfloat16
            v = v.contiguous().bfloat16()
            # v = v.to(torch.bfloat16)
            assert v.dtype == torch.bfloat16
            w = w.contiguous().bfloat16()
            # w = w.to(torch.bfloat16)
            assert w.dtype == torch.bfloat16
            u = u.contiguous().bfloat16()
            # u = u.to(torch.bfloat16)
            assert u.dtype == torch.bfloat16

            assert args.head_size_a == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ctx.save_for_backward(r, k, v, w, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, w, u, y)
            return y
    @staticmethod
    def backward(ctx, gy): # backward: gy => gr, gk, gv, gw, gu
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, w, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu) # return gradients for r,k,v,w,u


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        
        x = x.view(B * T, C)
        if self.ln_x.weight is not None:
            
            # 转为bfloat16，与x一致
            # 用.data.to()确保转换底层数据类型，而不是新建nn.Parameter
            # 默认requires_grad=True，不会丢失梯度
            self.ln_x.weight = nn.Parameter(self.ln_x.weight.data.to(x.dtype))
            
        if self.ln_x.bias is not None:
            self.ln_x.bias = nn.Parameter(self.ln_x.bias.data.to(x.dtype))
            
        x = self.ln_x(x).view(B, T, C)
        
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)

        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)
    
########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

########################################################################################################
# RWKV Block
########################################################################################################

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
        self.att = RWKV_Tmix_x060(config, layer_id)
        self.ffn = RWKV_CMix_x060(config, layer_id)
        
    def forward(self, x):

        if self.layer_id == 0:
            x = self.ln0(x)
        
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x










class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # module.weight.data.fill_(.01)  # KL: Adapter change


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config, index) for index in range(config.n_layer)])  #, scale=True
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(self, inputs_embeds=None):
        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class DecisionRWKV6(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            remove_act_embs=False,
            **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            dim_att = hidden_size,
            dim_ffn = int((hidden_size * 3.5) // 32 * 32),
            ctx_len = max_length,
            head_size_a = 64, # don't change
            head_size_divisor = 8, # don't change
            remove_act_embs=remove_act_embs,
            # max_length=max_length,
            **kwargs
        )

        #self.env_name = config.env_name

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        self.remove_act_embs = remove_act_embs

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # time embeddings are treated similar to positional embeddings
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        if not self.remove_act_embs:
            action_embeddings = self.embed_action(actions) + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if self.remove_act_embs:
            num_token_type = 2
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
        else:
            num_token_type = 3
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x = self.transformer(inputs_embeds=stacked_inputs)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, num_token_type, self.hidden_size).permute(0, 2, 1, 3)

        state_reps = x[:,1]
        action_preds = self.predict_action(state_reps)  # predict next action given state
        return action_preds

    def get_action(self, states, actions, returns_to_go, timesteps):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:,-self.max_length:]
        actions = actions[:,-self.max_length:]
        returns_to_go = returns_to_go[:,-self.max_length:]
        timesteps = timesteps[:,-self.max_length:]

        # pad all tokens to sequence length
        states = torch.cat(
            [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
            dim=1).to(dtype=torch.float32)
        actions = torch.cat(
            [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions],
            dim=1).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
            dim=1).to(dtype=torch.float32)
        timesteps = torch.cat(
            [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
            dim=1).to(dtype=torch.long)

        action_preds = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[0,-1]
