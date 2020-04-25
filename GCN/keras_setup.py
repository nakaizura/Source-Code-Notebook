'''
Created on Apr 25, 2020
@author: nakaizura
'''

from setuptools import setup
from setuptools import find_packages

#Setuptools是用于编译、分发和安装 python 包的一个工具
#特别是在包依赖问题场景下非常有用，它是一个强大的包管理工具

'''
setup.py适合一键打包安装
setup函数包括：

--name 包名称：生成的egg名称
--version包版本：生成egg包的版本号
--description 程序的简单描述：这个是keras版本的GCN
--author 作者
--author_email 作者的邮箱地址
--url 程序的官网地址
--download_url 程序的下载地址，如果有
--license 程序的授权信息：MIT....
--install_requires 安装依赖。主要是keras
--extras_require 其他的依赖。导入数据需要的json和h5py模块
--package_data 告诉setuptools哪些目录下的文件被映射到哪个源码包，都在README目录中。
--find_packages() 和setup.py同一目录下搜索各个含有 init.py的包，用于增加packages参数。

'''

setup(name='kegra',
      version='0.0.1',
      description='Deep Learning on Graphs with Keras',
      author='Thomas Kipf',
      author_email='thomas.kipf@gmail.com',
      url='https://tkipf.github.io',
      download_url='...',
      license='MIT',
      install_requires=['keras'],
      extras_require={
          'model_saving': ['json', 'h5py'],
      },
      package_data={'kegra': ['README.md']},
      packages=find_packages())
