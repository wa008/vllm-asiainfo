# inside `setup.py` file
from setuptools import setup

setup(name='vllm_asiainfo',
      version='0.1',
      packages=['vllm_asiainfo'],
      entry_points={
          "vllm.platform_plugins": ["register_asiainfo = vllm_asiainfo:register"],
      })

