from functools import partial

from .multiagentenv import MultiAgentEnv
from .matrix_game.cts_matrix_game import Matrixgame as CtsMatrix
from .particle import Particle
from .mamujoco import ManyAgentAntEnv, ManyAgentSwimmerEnv, MujocoMulti
from smac.env import MultiAgentEnv, StarCraft2Env
from .firefighters.firefighters import FireFightersEnv  # 注意导入路径

''' 
###################  原始版本  #######################
def env_fn(env, **kwargs) -> MultiAgentEnv:
    #env_args = kwargs.get("env_args", {})
    return env(**kwargs)
'''
#------------------新改的-------------------
def env_fn(env, **kwargs) -> MultiAgentEnv:
    # 提取env_args，如果不存在则使用空字典
    env_args = kwargs.pop('env_args', {})
    
    # 移除其他可能冲突的参数
    args = kwargs.pop('args', None)
    
    # 获取环境初始化方法的签名
    import inspect
    env_init_params = inspect.signature(env.__init__).parameters
    
    # 过滤参数，只保留环境初始化方法接受的参数
    filtered_kwargs = {k: v for k, v in {**env_args, **kwargs}.items() 
                       if k in env_init_params or env_init_params.get('kwargs')}
    
    # 如果scenario_dict在参数中，确保传递
    if 'scenario_dict' in env_args:
        filtered_kwargs['scenario_dict'] = env_args['scenario_dict']
    
    # 初始化环境
    env_instance = env(**filtered_kwargs)
    
    return env_instance


REGISTRY = {}
REGISTRY["cts_matrix_game"] = partial(env_fn, env=CtsMatrix)
REGISTRY["particle"] = partial(env_fn, env=Particle)
REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
REGISTRY["manyagent_swimmer"] = partial(env_fn, env=ManyAgentSwimmerEnv)
REGISTRY["manyagent_ant"] = partial(env_fn, env=ManyAgentAntEnv)
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["firefighters"] = partial(env_fn, env=FireFightersEnv)