from setuptools import find_packages, setup

setup(name='opytimizer',
      version='1.0.0',
      description='Meta-Inspired Python Optimizer',
      author='Gustavo Rosa',
      author_email='gth.rosa@uol.com.br',
      url='https://github.com/gugarosa/opytimizer',
      license='Apache-2.0',
      install_requires=['numpy>=1.13.3',
                        'py_expression_eval>=0.3.4',
                        'pylint>=1.7.4',
                        'pytest>=3.2.3',
                       ],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                   ],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache-2.0 License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
