from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import os
from model_config import MODEL_SPECS 
from replay_buffer import ReplayBuffer
import numpy as np
# This model is taken from: https://github.com/jaiwei98/MobileNetV4-pytorch/

def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
    ) -> int:
    """
    This function is copied from here 
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    
    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False, squeeze_excitation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_excitation:
            self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, 
            inp, 
            oup, 
            start_dw_kernel_size, 
            middle_dw_kernel_size, 
            middle_dw_downsample,
            stride,
            expand_ratio
        ):
        """An inverted bottleneck block with optional depthwises.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        """
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:            
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)
        
        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)
        
    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x

class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides, dw_kernel_size=3, dropout=0.4):
        """Multi Query Attention with spatial downsampling.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        3 parameters are introduced for the spatial downsampling:
        1. kv_strides: downsampling factor on Key and Values only.
        2. query_h_strides: vertical strides on Query only.
        3. query_w_strides: horizontal strides on Query only.

        This is an optimized version.
        1. Projections in Attention is explict written out as 1x1 Conv2D.
        2. Additional reshapes are introduced to bring a up to 3x speed up.
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(inp)
        self._query_proj = conv_2d(inp, num_heads*key_dim, 1, 1, norm=False, act=False)
        
        if self.kv_strides > 1:
            self._key_dw_conv = conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False)
            self._value_dw_conv = conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False)
        self._key_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)
        self._value_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)

        self._output_proj = conv_2d(num_heads*key_dim, inp, 1, 1, norm=False, act=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_length, _, _ = x.size()
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = F.avg_pool2d(self.query_h_stride, self.query_w_stride)
            q = self._query_downsampling_norm(q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        px = q.size(2)
        q = q.view(batch_size, self.num_heads, -1, self.key_dim) # [batch_size, num_heads, seq_length, key_dim]

        if self.kv_strides > 1:
            k = self._key_dw_conv(x)
            k = self._key_proj(k)
            v = self._value_dw_conv(x)
            v = self._value_proj(v)          
        else:
            k = self._key_proj(x)
            v = self._value_proj(x)
        k = k.view(batch_size, 1, self.key_dim, -1) # [batch_size, 1, key_dim, seq_length]
        v = v.view(batch_size, 1, -1, self.key_dim) # [batch_size, 1, seq_length, key_dim]

        # calculate attn score
        attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, dim=-1)

        context = torch.matmul(attn_score, v)
        context = context.view(batch_size, self.num_heads * self.key_dim, px, px)
        output = self._output_proj(context)
        return output

class MNV4LayerScale(nn.Module):
    def __init__(self, inp, init_value):
        """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        
        As used in MobileNetV4.

        Attributes:
            init_value (float): value to initialize the diagonal matrix of LayerScale.
        """
        super().__init__()
        self.init_value = init_value
        self._gamma = nn.Parameter(self.init_value * torch.ones(inp, 1, 1))
    
    def forward(self, x):
        return x * self._gamma

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
            self, 
            inp,
            num_heads, 
            key_dim,  
            value_dim, 
            query_h_strides, 
            query_w_strides, 
            kv_strides,
            use_layer_scale,
            use_multi_query, 
            use_residual = True
        ):
        super().__init__()
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual

        self._input_norm = nn.BatchNorm2d(inp)
        if self.use_multi_query:
            self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
                inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides
            )
        else:
            self.multi_head_attention = nn.MultiheadAttention(inp, num_heads, kdim=key_dim)
        
        if self.use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4LayerScale(inp, self.layer_scale_init_value) 
    
    def forward(self, x):
        # Not using CPE, skipped
        # input norm
        shortcut = x
        x = self._input_norm(x)
        # multi query
        if self.use_multi_query:
            x = self.multi_query_attention(x)
        else:
            x = self.multi_head_attention(x, x)
        # layer scale
        if self.use_layer_scale:
            x = self.layer_scale(x)
        # use residual
        if self.use_residual:
            x = x + shortcut
        return x

def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"convbn_{i}", conv_2d(**args))
    elif block_names == "uib":
        schema_ =  ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride', 'expand_ratio', 'mhsa']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            mhsa = args.pop("mhsa") if "mhsa" in args else 0
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
            if mhsa:
                mhsa_schema_ = [
                    "inp", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides", 
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args = dict(zip(mhsa_schema_, [args['oup']] + (mhsa)))
                layers.add_module(f"mhsa_{i}", MultiHeadSelfAttentionBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers

class ActorNetwork(nn.Module):
    def __init__(self, alpha, n_actions=1, rp_noise = 1e-6, fc1_dims=1280, fc2_dims=512,dir="tmp/"):
        super(ActorNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = os.path.join(dir, "actor_ppo")
        self.model = "MobileNetV4ConvSmall"
        self.spec = MODEL_SPECS[self.model]
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.rp_noise = rp_noise
        self.conv0 = build_blocks(self.spec['conv0'])
        # layer1
        self.layer1 = build_blocks(self.spec['layer1'])
        # layer2
        self.layer2 = build_blocks(self.spec['layer2'])
        # layer3
        self.layer3 = build_blocks(self.spec['layer3'])
        # layer4
        self.layer4 = build_blocks(self.spec['layer4'])
        # layer5   
        self.layer5 = build_blocks(self.spec['layer5'])
        self.fc1 = nn.Linear(fc1_dims, fc2_dims)
        # self.fc2 = nn.Linear(fc2_dims, n_actions)
        # self.softmax = nn.Softmax(dim = -1)
        self.fc_mean = nn.Linear(fc2_dims, n_actions)
        self.fc_log_var = nn.Linear(fc2_dims, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-3)
        self.dropout = nn.Dropout(p=0.4)
        self.CNNdropout = nn.Dropout(p=0.1)
        self.to(self.device)
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.CNNdropout(x5)
        x6 = nn.functional.adaptive_avg_pool2d(x6, 1)
        x6 = x6.reshape(x6.size(0), -1)
        x7 = F.tanh(self.fc1(x6))
        x7 = self.dropout(x7)
        # x6 = self.fc1(x6)
        mean = (self.fc_mean(x7))
        var = F.softplus(self.fc_log_var(x7)) # could replace with clamp to make it faster
        dist = torch.distributions.Normal(mean, var)
        # return [x1, x2, x3, x4, x5, x6, x7]
        return dist
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_dir)
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_dir))
class DoubleQCritic(nn.Module):
    def __init__(self, lr, action_dim = 1):
        super(DoubleQCritic, self).__init__()
        self.Q1_target = QNetwork(alpha = lr, action_dim = action_dim)
        self.Q2 = QNetwork(alpha = lr, action_dim = action_dim)
    def forward(self, state, action):
        sa = torch.cat([state,action], -1)
        q1 = self.Q1_target(sa)
        q2 = self.Q2(sa)
        return q1,q2
    def saveCheckpoints(self):
        self.Q1_target.save_checkpoint()
        self.Q2.save_checkpoint()
    def loadCheckpoints(self):
        self.Q1_target.load_checkpoint()
        self.Q2.load_checkpoint()
class QNetwork(nn.Module):
    def __init__(self, alpha, action_dim = 1, fc1_dims=1280, fc2_dims=512, dir = "tmp/"):
        super(QNetwork, self).__init__()
        self.checkpoint_dir = os.path.join(dir, "critic_ppo")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = "MobileNetV4ConvSmall"
        self.spec = MODEL_SPECS[self.model]
        self.conv0 = build_blocks(self.spec['conv0'])
        # layer1
        self.layer1 = build_blocks(self.spec['layer1'])
        # layer2
        self.layer2 = build_blocks(self.spec['layer2'])
        # layer3
        self.layer3 = build_blocks(self.spec['layer3'])
        # layer4
        self.layer4 = build_blocks(self.spec['layer4'])
        # layer5   
        self.layer5 = build_blocks(self.spec['layer5'])
        self.fc1 = nn.Linear(fc1_dims, fc2_dims)
        self.fc2 = nn.Linear(fc2_dims, 1)
     
        # self.fc_mean = nn.Linear(1280, 1)
        # self.fc_variance = nn.Linear(1280, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-3)
        self.dropout = nn.Dropout(p=0.4)
        self.CNNdropout = nn.Dropout(p=0.1)
        self.to(self.device)
    def forward(self, state, action):
        x = torch.cat([state,action], 1)
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = nn.functional.adaptive_avg_pool2d(x5, 1)
        x6 = x6.reshape(x6.size(0), -1)
        x6 = self.CNNdropout(x6)
        x7 = F.tanh(self.fc1(x6))
        x7 = self.dropout(x7)
        x8 = self.fc2(x7)
        return x8
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_dir)
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_dir))
class Agent:
    def __init__(self, n_actions, gamma=0.99, alpha = 0.0001, rp_noise = 1e-6, polyak = 0.99, batch_size = 64, n_epochs = 10):
        self.gamma = gamma
        self.n_actions = n_actions
        self.alpha = alpha 
        self.rp_noise = rp_noise
        self.polyak = polyak
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.actor= ActorNetwork(n_actions, alpha)
        self.critic = DoubleQCritic(alpha + 0.0001)
        self.critic_target = DoubleQCritic(alpha + 0.0001)
        self.memory = ReplayBuffer(batch_size=64, buffer_size=1000000, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), state_dim=(3,224,224))
    def store_memory(self, state, action, log_probs, val, reward, done):
        self.memory.store_memory(state,action, log_probs, val, reward,done)
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.saveCheckpoints()
        self.critic_target.saveCheckpoints()
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.loadCheckpoints()
        self.critic_target.loadCheckpoints()
    def update_targets(self):
        target_params = self.critic_target.Q1_target.parameters()
        source_params = self.critic.Q1_target.parameters()
        for target,source in zip(target_params, source_params):
            target.data.copy_((1-self.polyak) * source.data + self.polyak * target.data)
        target_params = self.critic_target.Q2_target.parameters()
        source_params = self.critic.Q2_target.parameters()
        for target,source in zip(target_params, source_params):
            target.data.copy_((1-self.polyak) * source.data + self.polyak * target.data)
    def sample_action(self, distribution):
        action = distribution.rsample()
        probs = distribution.log_prob(action).sum(dim=-1)
        action = F.tanh(action)
        probs -= torch.log(1-action.pow(2) + self.rp_noise)
        probs = probs.sum(1, keepdim=True)
        return action, probs
    def choose_action(self, state):
        if type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        dist = self.actor(state)
        val = self.critic(state)
        action, probs = self.sample_action(dist)

        action = torch.squeeze(action).item()
        probs = torch.squeeze(probs).item()
        value = torch.squeeze(val).item()
        
        return action, probs, value
    def learn(self):
        for _ in range(self.n_epochs):
            state, action, reward, next_state, done= self.memory.sample()
            value1, value2 = self.critic(state, action)
            value1 = value1.view(-1)
            value2 = value2.view(-1)
            dist = self.actor(next_state)
            next_action,log_probs = self.sample_action(dist)
            next_value1, next_value2 = self.critic_target(next_state, next_action) 
            next_value1 = next_value1.view(-1)
            next_value2 = next_value2.view(-1)
            log_probs = log_probs.view(-1)
            self.critic.Q1_target.optimizer.zero_grad()
            self.critic.Q2.optimizer.zero_grad()
            self.actor.optimizer.zero_grad()
            critic_loss1 = 0.5 * F.mse_loss(value1, reward + self.gamma*(1-done) *torch.min(next_value1, next_value2))
            critic_loss2 = 0.5 * F.mse_loss(value2, reward + self.gamma*(1-done) *torch.min(next_value1, next_value2))
            critic_loss = critic_loss1 + critic_loss2
            actor_loss = torch.mean((self.alpha * log_probs) - torch.min(value1, value2))
            critic_loss.backward()
            actor_loss.backward()
            self.critic.Q1_target.optimizer.step()
            self.critic.Q2.optimizer.step()
            self.actor.optimizer.step()
            self.update_targets()

# # net = Agent(4)
# # x = torch.rand(1, 3, 224, 224)
# # print(net.choose_action(x))
# net = ActorNetwork(4, 0.001)
# print(net.layer5)
# for name in (net.layer5.parameters()):
#     print(name.size())
# # for i in y:
# #     print(i.shape)
# # print(y[-1])