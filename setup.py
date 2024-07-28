#!/usr/bin/env python

from setuptools import setup

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='marl_experiments',
      version='0.0.1',
      description='Multi-Agent Experiments',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Joe Miceli, Chi-Hui Lin',
      author_email='joe.miceli@colorado.edu',
      url='https://github.com/HIRO-group/marl-experiments',
      download_url='',
      keywords=['Multi-agent RL', 'AI', 'Reinforcement Learning', 'Human Agent Collaboration'],
      packages=['marl_utils', 'rl_core', 'sumo_utils', 'sumo_utils.sumo_custom'],
      package_dir={
          'marl_utils': 'marl_utils',
          'rl_core': 'rl_core',
          'sumo_utils': 'sumo_utils',
          'sumo_utils.sumo_custom': 'sumo_utils/sumo_custom'
      },
      package_data={
        'marl_experiments' : [
          'package/*.pickle'
        ],
      },
      install_requires=[
        'sumo_rl',
        'torch',
        'pettingzoo'
      ],
      tests_require=['pytest']
    )