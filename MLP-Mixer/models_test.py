from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from vit_jax import models
from vit_jax.configs import models as config_lib

# 这个py可以用来测试一下模型的搭建

# 这里提供了多种Vision Transformer配置，如ViT等
MODEL_SIZES = {
    'ViT-B_16': 86_567_656,
    'R50+ViT-B_16': 98_659_112,
    'ViT-B_32': 88_224_232,
    'R26+ViT-B_32': 101_383_976,
    'ViT-L_16': 304_326_632,
    'ViT-L_32': 306_535_400,
    'R50+ViT-L_32': 328_994_856,
    'ViT-H_14': 632_045_800,
    'Mixer-B_16': 59_880_472,
    'Mixer-L_16': 208_196_168,
}


class ModelsTest(parameterized.TestCase):

  @parameterized.parameters(*list(MODEL_SIZES.items()))
  def test_can_instantiate(self, name, size):
    rng = jax.random.PRNGKey(0)
    config = config_lib.MODEL_CONFIGS[name] #模型配置
    model_cls = models.VisionTransformer if 'ViT' in name else models.MlpMixer #调用模型
    model = model_cls(num_classes=1_000, **config) #模型得到cls，用于图片分类1000
    
    inputs = jnp.ones([2, 224, 224, 3], jnp.float32) #随便定义一个input图片
    variables = model.init(rng, inputs, train=False) #初始化
    outputs = model.apply(variables, inputs, train=False) #得到模型输出
    self.assertEqual((2, 1000), outputs.shape)
    param_count = sum(p.size for p in jax.tree_flatten(variables)[0]) #计算参数
    self.assertEqual(
        size, param_count,
        f'Expected {name} to have {size} params, found {param_count}.')


if __name__ == '__main__':
  absltest.main()
