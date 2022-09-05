from typing import Any

import einops
import flax.linen as nn
import jax.numpy as jnp


class MlpBlock(nn.Module):
  mlp_dim: int

  @nn.compact
  def __call__(self, x): #这部分和上图的右侧MLP一致
    y = nn.Dense(self.mlp_dim)(x) #MLP+GELU+MLP
    y = nn.gelu(y)
    return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
  """Mixer block layer."""
  tokens_mlp_dim: int
  channels_mlp_dim: int

  @nn.compact
  def __call__(self, x): #这一部分和上面的公式中一致
    y = nn.LayerNorm()(x)
    y = jnp.swapaxes(y, 1, 2)
    y = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y) #token MLP
    y = jnp.swapaxes(y, 1, 2)
    x = x + y #残差部分
    y = nn.LayerNorm()(x)
    return x + MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y) #channel MLP


class MlpMixer(nn.Module):
  """Mixer architecture."""
  patches: Any
  num_classes: int
  num_blocks: int
  hidden_dim: int
  tokens_mlp_dim: int
  channels_mlp_dim: int

  @nn.compact
  def __call__(self, inputs, *, train): #把各组件搭起来
    del train
    x = nn.Conv(self.hidden_dim, self.patches.size,
                strides=self.patches.size, name='stem')(inputs)
    x = einops.rearrange(x, 'n h w c -> n (h w) c')
    for _ in range(self.num_blocks):
      x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
    x = nn.LayerNorm(name='pre_head_layer_norm')(x)
    x = jnp.mean(x, axis=1)
    return nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros,
                    name='head')(x)
