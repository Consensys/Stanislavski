# Stanislavski
pow gym environment

Implemenation of a proof of work algorithm powered by wittgenstein that can be run as a GYM environement and used for Reinforcement learning. 

## Dependencies

To get started you will need to install some dependencies

```
pip3 install gym
pip3 install Cython
pip3 install pyjnius

```

You will also need to install some wittgenstein files:
If you don't have the Wittgenstein repo already

```
git clone https://github.com/ConsenSys/wittgenstein.git
cd wittgestein
gradle clean shadowJar
```

This will allow you to call and run the Java code from python by creating a set of Jar files that will be . accessed through the pyjnius library.

## Configuring path

You need to setup the path to the to the JAR files in your computer in the *pow_env.py* file by changing the path in the jnius_config.set_classpath() :

```
import jnius_config
  jnius_config.set_classpath('.', './build/libs/wittgenstein-all.jar')
  from jnius import autoclass
  p = autoclass('net.consensys.wittgenstein.protocols.ethpow.ETHMinerAgent').create(0.25)
  p.init()
```

## Setup GYM_POW

Once you have run all the steps above go to the root folder where you see the gym_pow folder and run:
```
pip3 install -e gym_pow
```

Now you can call the environment and use any model you find suitable to train your agent. You can . build your pow_gym environment by using:

```python
import gym
import gym_pow

env = gym.make('pow-v0')
```
