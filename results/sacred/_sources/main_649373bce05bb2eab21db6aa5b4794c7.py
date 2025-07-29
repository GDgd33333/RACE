import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import datetime
from run import run

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
th.set_num_threads(cpu_num)

SETTINGS['CAPTURE_MODE'] = "no" # set to "no" if you want to see stdout/stderr in console
SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
logger = get_logger()

ex = Experiment("pymarl",save_git_info=False)
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # -------------------------------------------------------
    # ------------------ 自动保存控制台输出 ------------------
    # -------------------------------------------------------
    config = config_copy(_config)  # 必须最先复制 config！

    # ------- 自动保存控制台输出（print + logger）-------
    log_base = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(log_base, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config['name']}_{now}"
    log_dir = os.path.join(log_base, run_id)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "log.txt")

    # 重定向 print 输出
    sys.stdout = open(log_file, 'w')
    sys.stderr = sys.stdout
    print(f"[INFO] Logging to: {log_file}")

    # 配置 logging 模块输出到文件
    import logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(name)s %(message)s", datefmt="%H:%M:%S")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    # 保存 config 到日志目录
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # -------------------------------------------------------
    # -------------------------------------------------------
    # -------------------------------------------------------
    # -------------------------------------------------------
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    """
    递归地更新字典。

    Args:
        d (dict): 要更新的原始字典。
        u (dict): 包含更新值的字典。

    Returns:
        dict: 更新后的字典。

    描述:
        递归地将更新字典 `u` 中的值合并到原始字典 `d` 中。
        如果 `u` 中的值是另一个字典，则会递归地调用此函数以更新嵌套字典。
        如果 `u` 中的值不是字典，则直接替换 `d` 中对应的值。
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    """
    深度复制给定的配置对象。

    Args:
        config (dict, list): 需要复制的配置对象，支持字典和列表类型。

    Returns:
        dict, list: 深度复制后的配置对象，类型与输入相同。

    """
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
