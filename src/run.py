import datetime
from functools import partial
from math import ceil
import numpy as np
import os
import pprint
import time
import threading
import torch as th
from gym import spaces
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from ea import mod_neuro_evo as utils_ne

# rl_to_evo 实现 将RL智能体的策略网络参数复制到EA个体的对应网络，完成从强化学习到进化算法的策略知识迁移。
def rl_to_evo(rl_agent, evo_net, index):
    # 遍历RL智能体和EA个体中第index个智能体的策略网络参数
    for target_param, param in zip(evo_net.agent_W[index].parameters(), rl_agent.agent_W[index].parameters()):
        target_param.data.copy_(param.data)  # 将RL网络的参数值复制到EA网络


def run(_run, _config, _log, pymongo_client=None):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    if pymongo_client is not None:
        print("Attempting to close mongodb client")
        pymongo_client.close()
        print("Mongodb client closed")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


import wandb

os.environ["WANDB_API_KEY"] = "2a3b6e7a8a2b7996077fc5439051f6cfeb6ff77d"
os.environ["WANDB_MODE"] = "disabled"
import time

'''
主要流程总结
1.初始化阶段:
    设置实验名称和WandB记录
    初始化环境runner
    配置数据格式方案和预处理
    初始化经验回放缓冲区
    创建进化算法实例
    初始化RL智能体和EA种群(如果使用EA)
    配置学习器

2.训练循环:
    定期执行EA评估和更新(如果使用EA):
    评估种群中每个个体
    执行进化算法操作(变异、交叉等)
    将最佳EA个体与RL智能体进行比较

3.常规RL训练:
    收集经验并存入缓冲区
    从缓冲区采样批次进行训练

4.定期测试评估:
    评估RL智能体和EA最佳个体的性能
    记录评估结果

5.策略同步:
    将RL策略同步到EA种群中表现较差的个体
    模型保存和日志记录

6.训练结束:
    关闭环境
    输出训练完成信息
'''


def run_sequential(args, logger):
    # 初始化WandB实验记录，创建唯一的实验名称
    # 示例数据: name = "RACE_VMM_0.5_Mutation_alpha_0.1_VFM_0.3_Map_env_8m_Use_EA_True"
    '''
    name = "RACE_VMM_" + str(args.state_alpha) + "_Mutation_alpha_" + str(args.frac) + "_VFM_" + str(
        args.EA_alpha) + "_Map_env_" + str(args.env_args["map_name"]) + "_Use_EA_" + str(args.EA)
    '''
    #-----------------------------------------------------------------
    #修改为兼容Firefighter的代码
    #-----------------------------------------------------------------
    name = "RACE_VMM_" + str(args.state_alpha) + "_Mutation_alpha_" + str(args.frac) + "_VFM_" + str(
        args.EA_alpha) + "_EnvConfig_" + str(args.env_args.get("map_name", args.env_args.get("scenario_dict", "unknown"))) + "_Use_EA_" + str(args.EA)
    our_wandb = wandb.init(project="RACE", name=name)  # 初始化WandB记录

    # Init runner so we can get env info
    # args.runner = "episode"表示使用回合制runner
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    if 'particle' not in args.env and "cts_matrix_game" not in args.env and "mujoco_multi" not in args.env:
        env_info = runner.get_env_info()
        '''
        确定神经网络输入/输出维度，如Actor网络的输入应为obs_shape=36。
        env_info: {
            'state_shape': 27,      # 全局状态维度
            'obs_shape': 36,        # 单个智能体观测维度
            'n_actions': 9,         # 离散动作数量
            'n_agents': 3,          # 智能体数量
            'episode_limit': 150,   # 最大回合步数
            'agent_features': ['health', 'energy/cooldown', 'rel_x', 'rel_y', 'shield'],  # 智能体特征
            'enemy_features': ['health', 'rel_x', 'rel_y', 'shield']                      # 敌方特征
        }
        '''
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(env_info)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        args.n_agents = env_info["n_agents"]        # 3
        args.n_actions = env_info["n_actions"]      # 9
        args.state_shape = env_info["state_shape"]  # 27
        args.obs_shape = env_info["obs_shape"]      # 36

        # 定义数据格式方案
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},   # actions存储实际执行的动作索引
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int}, # avail_actions是长度为9的二进制向量，表示哪些动作可用（如[1, 0, 1, ...]表示动作0和2可用）。
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])  # 动作one-hot编码
        }
    else:
        # 处理连续动作空间的环境
        env_info = runner.get_env_info()
        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.action_spaces = env_info["action_spaces"]
        args.actions_dtype = env_info["actions_dtype"]
        args.normalise_actions = env_info.get("normalise_actions", False)  # if true, action vectors need to sum to one

        # 创建动作空间转换张量
        ttype = th.FloatTensor if not args.use_cuda else th.cuda.FloatTensor
        mult_coef_tensor = ttype(args.n_agents, args.n_actions)
        action_min_tensor = ttype(args.n_agents, args.n_actions)

        # 填充转换系数张量
        if all([isinstance(act_space, spaces.Box) for act_space in args.action_spaces]):
            for _aid in range(args.n_agents):
                for _actid in range(args.action_spaces[_aid].shape[0]):
                    _action_min = args.action_spaces[_aid].low[_actid]
                    _action_max = args.action_spaces[_aid].high[_actid]
                    mult_coef_tensor[_aid, _actid] = np.asscalar(_action_max - _action_min)
                    action_min_tensor[_aid, _actid] = np.asscalar(_action_min)
        elif all([isinstance(act_space, spaces.Tuple) for act_space in args.action_spaces]):
            for _aid in range(args.n_agents):
                for _actid in range(args.action_spaces[_aid].spaces[0].shape[0]):
                    _action_min = args.action_spaces[_aid].spaces[0].low[_actid]
                    _action_max = args.action_spaces[_aid].spaces[0].high[_actid]
                    mult_coef_tensor[_aid, _actid] = np.asscalar(_action_max - _action_min)
                    action_min_tensor[_aid, _actid] = np.asscalar(_action_min)
                for _actid in range(args.action_spaces[_aid].spaces[1].shape[0]):
                    _action_min = args.action_spaces[_aid].spaces[1].low[_actid]
                    _action_max = args.action_spaces[_aid].spaces[1].high[_actid]
                    tmp_idx = _actid + args.action_spaces[_aid].spaces[0].shape[0]
                    mult_coef_tensor[_aid, tmp_idx] = np.asscalar(_action_max - _action_min)
                    action_min_tensor[_aid, tmp_idx] = np.asscalar(_action_min)

        # 存储转换参数
        args.actions2unit_coef = mult_coef_tensor
        args.actions2unit_coef_cpu = mult_coef_tensor.cpu()
        args.actions2unit_coef_numpy = mult_coef_tensor.cpu().numpy()
        args.actions_min = action_min_tensor
        args.actions_min_cpu = action_min_tensor.cpu()
        args.actions_min_numpy = action_min_tensor.cpu().numpy()

        # 定义动作空间转换函数
        def actions_to_unit_box(actions):
            # 将动作从原始空间转换到[0,1]单位空间
            if isinstance(actions, np.ndarray):
                return args.actions2unit_coef_numpy * actions + args.actions_min_numpy
            elif actions.is_cuda:
                return args.actions2unit_coef * actions + args.actions_min
            else:
                return args.args.actions2unit_coef_cpu * actions + args.actions_min_cpu

        def actions_from_unit_box(actions):
            # 将动作从[0,1]单位空间转换回原始空间
            if isinstance(actions, np.ndarray):
                return th.div((actions - args.actions_min_numpy), args.actions2unit_coef_numpy)
            elif actions.is_cuda:
                return th.div((actions - args.actions_min), args.actions2unit_coef)
            else:
                return th.div((actions - args.actions_min_cpu), args.actions2unit_coef_cpu)

        # make conversion functions globally available
        args.actions2unit = actions_to_unit_box
        args.unit2actions = actions_from_unit_box

        # 定义连续动作空间的数据格式
        action_dtype = th.long if not args.actions_dtype == np.float32 else th.float
        if all([isinstance(act_space, spaces.Box) for act_space in args.action_spaces]):
            actions_vshape = 1 if not args.actions_dtype == np.float32 else max(
                [i.shape[0] for i in args.action_spaces])
        elif all([isinstance(act_space, spaces.Tuple) for act_space in args.action_spaces]):
            actions_vshape = 1 if not args.actions_dtype == np.float32 else \
                max([i.spaces[0].shape[0] + i.spaces[1].shape[0] for i in args.action_spaces])
        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (actions_vshape,), "group": "agents", "dtype": action_dtype},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.n_agents
        }

        if not args.actions_dtype == np.float32:
            preprocess = {
                "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
            }
        else:
            preprocess = {}

    # 初始化经验回放缓冲区
    buffer = ReplayBuffer(scheme, groups, args.buffer_size,
                          env_info["episode_limit"] + 1 if args.runner_scope == "episodic" else 2,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    '''
        buffer: ReplayBuffer. 0/5000 episodes.q 
        Keys:dict_keys(['state', 'obs', 'actions', 'avail_actions', 'reward', 'terminated', 'actions_onehot', 'filled']) 
        Groups:dict_keys(['agents'])
    '''
    # 初始化进化算法
    evolver = utils_ne.SSNE(args)  # SSNE: 稳态神经进化算法

    # Setup multiagent controller here
    if args.EA:  # 如果使用进化算法
        pop = []  # 种群列表
        mac = mac_REGISTRY["RL_" + args.mac](buffer.scheme, groups, args)  # 主RL智能体
        for i in range(5):  # 示例: 种群大小为5
            pop.append(mac_REGISTRY["EA_" + args.mac](buffer.scheme, mac.agent_SR, groups, args))  # 进化算法智能体
        print("===================================================================================")
        print("pop:", pop)
        print("===================================================================================")
        '''
        pop: [<controllers.EA_basic_controller.Gen_BasicMAC object at 0x746bb73d9160>, <controllers.EA_basic_controller.Gen_BasicMAC object at 0x746bb72e0040>, <controllers.EA_basic_controller.Gen_BasicMAC object at 0x746bb72e0400>, <controllers.EA_basic_controller.Gen_BasicMAC object at 0x746bb72e0610>, <controllers.EA_basic_controller.Gen_BasicMAC object at 0x746bb72e0820>]
        '''
    else:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)  # 仅使用RL智能体
        pop = []  # 空种群
    fitness = []  # 存储种群中每个个体的适应度

    # Give runner the scheme
    # 配置runner
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 初始化学习器
    if args.EA:
        learner = le_REGISTRY[args.learner + "_EA"](mac, buffer.scheme, logger, args)  # 支持EA的学习器
        '''
        learner: <learners.EA_facmac_learner_discrete.EA_FACMACDiscreteLearner object at 0x746bb72cdfa0>
        '''
    else:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)  # 普通学习器

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # 开始训练循环
    episode = 0  # 记录总回合数
    last_test_T = - args.test_interval - 1  # 上次测试的时间步
    last_log_T = 0  # 上次记录日志的时间步
    model_save_time = 0  # 上次保存模型的时间步

    start_time = time.time()  # 训练开始时间
    last_time = start_time  # 用于计算剩余时间

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        if getattr(args, "runner_scope", "episodic") == "episodic":  # 回合制训练模式
            # 如果使用EA且达到EA更新频率，则进行EA进化
            if args.EA and runner.t_env > args.start_timesteps and episode % args.EA_freq == 0:
                fitness = []  # 重置适应度列表
                # 评估种群中每个个体
                for ea_mac in pop:
                    episode_batch, episode_return, _ = runner.run(ea_mac, test_mode=False, EA=True, learner=learner)
                    fitness.append(episode_return)  # 示例: [120.5, 115.3, 125.7, 110.2, 118.6]
                    # print("EA ", episode_batch.batch_size)
                    buffer.insert_episode_batch(episode_batch)  # 将EA经验存入缓冲区

                # 评估RL智能体
                elite_index = np.argmax(fitness)  # 示例: 2 (最佳个体索引)
                episode_batch, _, _ = runner.run(mac, EA=False, test_mode=False)
                # print("RL ", episode_batch.batch_size)
                buffer.insert_episode_batch(episode_batch)  # 将RL经验存入缓冲区

                print("EA evolution start ...")

                # 执行进化算法更新 （SAME=0）
                if args.SAME:  # 如果所有智能体共享策略
                    elite_index = evolver.epoch(pop, fitness, 0, agent_level=True)
                else:  # 每个智能体独立进化   （是这个）
                    for agent_index in range(args.n_agents):
                        elite_index = evolver.epoch(pop, fitness, agent_index, agent_level=True)
                    # elite_index = evolver.epoch(pop, fitness, 0, agent_level=True)
                print("EA evolution end.")
            else:
                # 不执行EA更新的常规训练
                fitness = [0, 0, 0, 0, 0]
                elite_index = 0
                episode_batch, _, _ = runner.run(mac, test_mode=False, EA=False, learner=learner)
                buffer.insert_episode_batch(episode_batch)

            # 确定训练重复次数(EA更新后多训练几次)
            if args.EA and runner.t_env > args.start_timesteps and episode % args.EA_freq == 0:
                repeat_num = 6
            else:
                repeat_num = 1

            # 执行训练步骤
            for i in range(repeat_num):
                start = time.time()
                if buffer.can_sample(args.batch_size) and (
                        buffer.episodes_in_buffer > getattr(args, "buffer_warmup", 0)):
                    episode_sample = buffer.sample(args.batch_size)  # 示例: batch_size=32

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()  # 获取实际填充的时间步
                    episode_sample = episode_sample[:, :max_ep_t]  # 截断批次

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)  # 确保数据在正确设备上

                    # 执行训练
                    if args.EA:
                        all_teams = []  # 合并EA种群和RL智能体
                        for index_n in range(args.pop_size):
                            all_teams.append(pop[index_n])
                        all_teams.append(mac)
                        learner.train(episode_sample, all_teams, runner.t_env, episode)
                    else:
                        learner.train(episode_sample, runner.t_env, episode)
                # print("train one batch time cost ", time.time()-start)
                # learner.train(episode_sample, runner.t_env, episode)
        elif getattr(args, "runner_scope", "episode") == "transition":
            fitness = []
            for ea_mac in pop:
                episode_batch, episode_return, _ = runner.run(ea_mac, test_mode=False, EA=True, buffer=buffer,
                                                              learner=learner, episode=episode)
                fitness.append(episode_return)
                # print("EA ", episode_batch.batch_size)
                buffer.insert_episode_batch(episode_batch)
            elite_index = np.argmax(fitness)
            episode_batch, _, _ = runner.run(mac, test_mode=False, EA=False, buffer=buffer, learner=learner,
                                             episode=episode)
            print("RL ", episode_batch.batch_size)
            buffer.insert_episode_batch(episode_batch)

            print("EA evolution start ...")
            for agent_index in range(args.n_agents):
                elite_index = evolver.epoch(pop, fitness, agent_index, agent_level=True)
            # elite_index = evolver.epoch(pop, fitness, 0, agent_level=True)
            print("EA evolution end.")

        else:
            raise Exception("Undefined runner scope!")

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)

        # 测试评估
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env

            # 初始化RL评估指标
            rl_eval_reward = 0       # RL智能体累计奖励
            rl_battle_win_rate = 0   # RL智能体胜率
            # 进行n_test_runs次测试取平均值
            for _ in range(n_test_runs):
                # 根据runner_scope选择不同的运行模式
                if getattr(args, "runner_scope", "episodic") == "episodic":
                    _, rl_tp_reward, tp_win = runner.run(mac, test_mode=True, learner=learner)
                elif getattr(args, "runner_scope", "episode") == "transition":
                    _, rl_tp_reward, tp_win = runner.run(mac, test_mode=True, buffer=buffer, learner=learner,
                                                         episode=episode)
                else:
                    raise Exception("Undefined runner scope!")
                # 累加单次测试结果
                rl_eval_reward += rl_tp_reward
                rl_battle_win_rate += tp_win
            # 计算平均表现
            rl_eval_reward = rl_eval_reward / n_test_runs
            rl_battle_win_rate = rl_battle_win_rate / n_test_runs
            print("RL eval ", rl_eval_reward, rl_battle_win_rate)

            # 如果启用了EA，评估EA种群表现
            if args.EA:
                ea_eval_reward = 0
                ea_battle_win_rate = 0
                last_test_T = runner.t_env

                # 测试EA种群中适应度最高的个体（精英）
                for _ in range(n_test_runs):
                    if getattr(args, "runner_scope", "episodic") == "episodic":
                        _, ea_tp_reward, tp_win = runner.run(pop[np.argmax(fitness)], test_mode=True, learner=learner)
                    elif getattr(args, "runner_scope", "episode") == "transition":
                        _, ea_tp_reward, tp_win = runner.run(pop[np.argmax(fitness)], test_mode=True, buffer=buffer,
                                                             learner=learner, episode=episode)
                    else:
                        raise Exception("Undefined runner scope!")

                    ea_eval_reward += ea_tp_reward
                    ea_battle_win_rate += tp_win

                # 计算EA精英平均表现
                ea_eval_reward = ea_eval_reward / n_test_runs
                ea_battle_win_rate = ea_battle_win_rate / n_test_runs
            else:
                # 未启用EA时设为默认值
                ea_eval_reward = 0
                ea_battle_win_rate = 0.0

            print("EA eval ", ea_eval_reward, ea_battle_win_rate)

            # 记录到WandB
            our_wandb.log(
                {'Test_RL_reward': rl_eval_reward, 'Test_EA_reward': ea_eval_reward,
                 'Test_best_reward': np.max([rl_eval_reward, ea_eval_reward]), 'Test_RL_won_rate': rl_battle_win_rate,
                 'Test_EA_won_rate': ea_battle_win_rate,
                 'Test_Best_won_rate': np.max([ea_battle_win_rate, rl_battle_win_rate]), "time_steps:": runner.t_env})

            if getattr(args, "testing_on", True):
                for _ in range(n_test_runs):
                    if getattr(args, "runner_scope", "episodic") == "episodic":
                        runner.run(mac, test_mode=True, learner=learner)
                    elif getattr(args, "runner_scope", "episode") == "transition":
                        runner.run(mac, test_mode=True,
                                   buffer=buffer,
                                   learner=learner,
                                   episode=episode)
                    else:
                        raise Exception("Undefined runner scope!")


        # EA->RL策略同步
        if args.EA and runner.t_env > args.start_timesteps and episode % args.EA_freq == 0:
            # Replace any index different from the new elite
            replace_index = np.argmin(fitness)   # 找到适应度最低的个体索引
            if replace_index == elite_index:   # 如果最差个体恰好是精英
                replace_index = (replace_index + 1) % args.pop_size   # 改为替换下一个个体
            for index in range(args.n_agents):
                rl_to_evo(mac, pop[replace_index], index)  #将 RL智能体（mac） 的策略参数复制到 EA种群中的目标个体（pop[replace_index]）。
            evolver.rl_policy = replace_index  #标记RL策略在EA种群中的位置（replace_index），用于后续统计（如跟踪RL策略是否被保留/淘汰）。
            print('Sync from RL --> Nevo')


        # 模型保存
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            # learner.save_models(save_path, args.unique_token, model_save_time)

            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
