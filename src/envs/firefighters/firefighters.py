from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from itertools import product
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from .multiagentenv import MultiAgentEnv


class Render(object):
    def __init__(self, figsize=(15, 15), dpi=48):
        self.figsize = figsize
        self.dpi = dpi
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.artists = []

    def add_artist(self, artist):
        self.artists.append(artist)

    def new_frame(self):
        self.fig.clear()
        self.ax = self.fig.gca()
        self.ax.clear()
        for artist in self.artists:
            artist.remove()
        self.artists = []
        self.ax.set_xlim(-0.1, MAP_SIZE + 1.1)
        self.ax.set_ylim(-0.1, MAP_SIZE + 1.1)
        self.ax.axis('off')

    def draw(self):
        for artist in self.artists:
            self.ax.add_artist(artist)
        self.canvas.draw()       # draw the canvas, cache the renderer
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        image = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return image

    def render(self, img):
        plt.figure(figsize=self.figsize)
        ax = plt.gca()
        ax.axis('off')
        plt.imshow(img)
        plt.show()


class Entity(object):
    def __init__(self, obj_id, ent_id, ent_type):
        self.obj_id = obj_id     # 初始化对象ID
        self.ent_id = ent_id     # 初始化实体ID
        self.ent_type = ent_type    # 初始化实体类型
        self.x = None  # 初始化x坐标，默认为None
        self.y = None  # 初始化y坐标，默认为None



class Building(Entity):
    def __init__(self, x, y, obj_id, ent_id, game_config,
                 burn_rate, init_health_range, fire_start_rate_range):
        # 调用父类 Building 的构造函数
        super(Building, self).__init__(obj_id, ent_id, ent_type='building')
        self.x = x  # 建筑在地图上的 x 坐标
        self.y = y  # 建筑在地图上的 y 坐标
        self.burn_rate = burn_rate        # 设置建筑的燃烧速率
        self.game_config = game_config    # 设置游戏配置
        rand = np.random.rand()        # 随机初始化建筑的生命值
        self.health = (init_health_range[0] * rand  # 生命值的最小值乘以随机数
                       + init_health_range[1] * (1 - rand))  # 生命值的最大值乘以 (1 - 随机数)
        self.fire_strength = 0.0        # 初始化火焰强度
        self.complete = (self.health == 1.0)        # 判断建筑是否完整（生命值是否为1.0）
        self.burned_down = (self.health == 0.0)        # 判断建筑是否烧毁（生命值是否为0.0）
        self.fire_decrement = 0.0        # 初始化火焰减少量
        self.build_delta = 0.0        # 初始化建造增量
        self.is_frozen = False        # 初始化建筑是否被冻结的状态
        self.fire_dec_set = set()        # 初始化火焰减少集合
        self.build_delta_set = set()        # 初始化建造增量集合

        # 随机初始化建筑下一次起火的速率
        self.my_fire_start_rate = np.random.randint(
            fire_start_rate_range[0],  # 起火速率范围的最小值
            fire_start_rate_range[1] + 1)  # 起火速率范围的最大值加1

        # 初始化到下一次起火的时间
        self.time_to_next_fire = self.my_fire_start_rate

        # 初始化自上次起火以来的时间
        self.time_since_last_fire = self.my_fire_start_rate - self.time_to_next_fire


    def reduce_fire(self, fire_speed, ent_id):
        """
        减少指定实体的火焰强度。
        Args:
            fire_speed (float): 火焰减少的速度。
            ent_id (int): 需要减少火焰的实体ID。
        Returns:
            None
        描述:
            该方法将指定实体的火焰强度按照给定的火焰减少速度减少，并将该实体的ID添加到火焰减少集合中。
            火焰减少的速度会根据游戏配置中的'fire_reduce_rate'进行调整。
        """
        self.fire_decrement += fire_speed * self.game_config['fire_reduce_rate']
        self.fire_dec_set.add(ent_id)

    def build(self, build_speed, ent_id):
        """
        构建建筑。
        Args:
            build_speed (float): 建设速度，一个浮点数。
            ent_id (int): 实体ID，一个整数。
        Returns:
            None
        该方法将建设速度 (`build_speed`) 与游戏配置中的基础建设速率 (`base_build_rate`) 相乘，
        并将结果累加到 `self.build_delta` 属性上。同时，将实体ID (`ent_id`) 添加到 `self.build_delta_set` 集合中。
        """
        self.build_delta += build_speed * self.game_config['base_build_rate']
        self.build_delta_set.add(ent_id)

    def freeze(self):
        # 冻结当前对象
        self.is_frozen = True

    def set_fire(self):
        """
        设置火的初始状态。

        这个函数用于初始化或重置火的状态，包括：
        - fire_strength: 火的强度，设置为1.0。
        - time_to_next_fire: 到下一次点火的时间，设置为my_fire_start_rate。
        - time_since_last_fire: 自上次点火以来的时间，设置为0。
        """
        self.fire_strength = 1.0
        self.time_to_next_fire = self.my_fire_start_rate
        self.time_since_last_fire = 0

    def step(self):
        """
        执行游戏的一步操作，更新火势、建筑健康值等状态，并返回相关信息。
        Returns:
        - dict: 包含火势、建筑等状态变化信息的字典。
        """
        info = {   # 初始化信息字典，记录各种状态的变化
            'fires_extinguished': 0,
            'buildings_completed': 0,
            'buildings_burned': 0,
            'buildings_health_delta': 0
        }

        # 根据火势减量的不同情况计算新的火势
        if self.fire_decrement == 0.0:
            # 如果火势减量为0，则根据游戏配置计算新的火势
            new_fire = self.game_config['fire_growth_rate'] * self.fire_strength
        else:
            # 否则，根据火势减量和其他条件计算新的火势
            if B_ID in self.fire_dec_set and G_ID in self.fire_dec_set:
                # 如果满足特定条件，增加火势减量
                self.fire_decrement += self.game_config['generalist_help_factor'] * self.game_config['fire_reduce_rate']
            new_fire = self.fire_strength - self.fire_decrement

        # 记录之前的火势
        prev_fire = self.fire_strength

        # 更新火势，确保火势在0到1之间
        self.fire_strength = min(1.0, max(0.0, new_fire))

        # 根据建筑增量的不同情况计算新的建筑健康值
        if F_ID in self.build_delta_set and G_ID in self.build_delta_set:
            # 如果满足特定条件，增加建筑增量
            self.build_delta += self.game_config['generalist_help_factor'] * self.game_config['base_build_rate']

        # 如果建筑未冻结，则根据火势减少建筑增量
        if not self.is_frozen:
            self.build_delta -= self.fire_strength * self.burn_rate * self.game_config['fire_burn_rate']

        # 计算新的建筑健康值
        new_health = self.health + self.build_delta

        # 记录之前的建筑健康值
        prev_health = self.health

        # 更新建筑健康值，确保健康值在0到1之间
        self.health = min(1.0, max(0.0, new_health))

        # 计算建筑健康值的真实变化量
        true_delta = self.health - prev_health

        # 如果之前的火势大于0且当前火势为0，则记录一次灭火
        if prev_fire > 0.0 and self.fire_strength == 0.0:
            info['fires_extinguished'] += 1

        # 如果之前的建筑健康值小于1且当前建筑健康值为1，则记录一次建筑完成
        if prev_health < 1.0 and self.health == 1.0:
            info['buildings_completed'] += 1

        # 如果之前的建筑健康值大于0且当前建筑健康值为0，则记录一次建筑烧毁
        if prev_health > 0.0 and self.health == 0.0:
            info['buildings_burned'] += 1
            self.fire_strength = 0.0

        # 如果火势减量集合不为空，则记录参与灭火的类型数量
        if len(self.fire_dec_set) > 0:
            info['types_per_firefight'] = len(self.fire_dec_set)
            info['types_per_firefight-count'] = 1

        # 如果建筑增量集合不为空，则记录参与建筑修复的类型数量
        if len(self.build_delta_set) > 0:
            info['types_per_build'] = len(self.build_delta_set)
            info['types_per_build-count'] = 1

        # 更新建筑健康值的变化量
        info['buildings_health_delta'] += true_delta

        # 重置火势减量和建筑增量
        self.fire_decrement = 0.0
        self.build_delta = 0.0

        # 重置建筑冻结状态
        self.is_frozen = False

        # 清空火势减量和建筑增量集合
        self.fire_dec_set = set()
        self.build_delta_set = set()

        # 更新建筑完成和烧毁状态
        self.complete = (self.health == 1.0)
        self.burned_down = (self.health == 0.0)

        # 根据建筑状态更新下次火灾时间和自上次火灾以来的时间
        if not (self.complete or self.burned_down):
            self.time_to_next_fire -= 1
            self.time_since_last_fire += 1
            if self.time_to_next_fire <= 0:
                self.set_fire()
        else:
            self.time_to_next_fire = 1000
            self.time_since_last_fire = 0

        # 返回信息字典
        return info



class FastBurnBuilding(Building):
    def __init__(self, x, y, obj_id, init_health_range, game_config):
        # 调用父类的构造函数
        super(FastBurnBuilding, self).__init__(
            x=x,                           # X坐标
            y=y,                           # Y坐标
            obj_id=obj_id,                 # 对象ID
            ent_id=0,                      # 实体ID（固定为0）
            game_config=game_config,       # 游戏配置
            burn_rate=1.0,                 # 燃烧速率（固定为1.0）
            init_health_range=init_health_range,  # 初始健康值范围
            fire_start_rate_range=(60, MAX_FIRE_START_RATE)  # 火灾起始速率范围（60，80）
        )



class SlowBurnBuilding(Building):
    def __init__(self, x, y, obj_id, init_health_range, game_config):
        # 调用父类的构造函数
        super(SlowBurnBuilding, self).__init__(
            x=x,                    # 坐标x
            y=y,                    # 坐标y
            obj_id=obj_id,          # 对象ID
            ent_id=1,               # 实体ID
            game_config=game_config,# 游戏配置
            burn_rate=0.5,          # 燃烧速率
            init_health_range=init_health_range, # 初始健康值范围
            fire_start_rate_range=(MIN_FIRE_START_RATE, 30) # 火灾起始速率范围（10，30）
        )



class Agent(Entity):
    def __init__(self, obj_id, ent_id, fire_speed, build_speed, can_freeze,
                 max_move_dist, sight_range):
        # 调用父类Agent的初始化方法，设置obj_id和ent_id，并指定实体类型为'agent'
        super(Agent, self).__init__(obj_id, ent_id, ent_type='agent')
        self.fire_speed = fire_speed          # 设置开火速度
        self.build_speed = build_speed        # 设置建造速度
        self.can_freeze = can_freeze          # 设置是否可以冻结
        self.max_move_dist = max_move_dist    # 设置最大移动距离
        self.sight_range = sight_range        # 设置视野范围
        self.prev_x = None              # 初始化上一次的x坐标位置为None
        self.prev_y = None              # 初始化上一次的y坐标位置为None
        self.freeze_actions = 0         # 初始化冻结动作计数器为0
        self.build_actions = 0          # 初始化建造动作计数器为0
        self.fire_actions = 0           # 初始化开火动作计数器为0
        self.last_action = None         # 初始化上一次执行的动作为None

#-------------------------------
#-------------消防员-------------
#------------------------------
class FireFighter(Agent):
    def __init__(self, obj_id):
        super(FireFighter, self).__init__(obj_id=obj_id,          # 对象ID
                                          ent_id=F_ID,            # 实体ID
                                          fire_speed=FF_F_SPEED,  # 消防员的灭火速度
                                          build_speed=FF_B_SPEED, # 消防员的建设速度
                                          can_freeze=False,       # 是否具有冻结能力
                                          max_move_dist=1,        # 消防员的最大移动距离，这里设置为1
                                          sight_range=None)       # 消防员的视野范围

#-------------------------------
#-------------建筑工-------------
#------------------------------
class Builder(Agent):
    def __init__(self, obj_id):
        super(Builder, self).__init__(obj_id=obj_id,
                                      ent_id=B_ID,
                                      fire_speed=B_F_SPEED,
                                      build_speed=B_B_SPEED,
                                      can_freeze=False,
                                      max_move_dist=1,
                                      sight_range=None)

#-------------------------------------------------------------------------------------
#-------------多面手：可冻结建筑、移动距离2、无直接灭火/建造能力但能增强其他智能体效果-------------
#-------------------------------------------------------------------------------------
class Generalist(Agent):
    def __init__(self, obj_id):
        super(Generalist, self).__init__(obj_id=obj_id,
                                         ent_id=G_ID,
                                         fire_speed=0.0,
                                         build_speed=0.0,
                                         can_freeze=True,
                                         max_move_dist=2,
                                         sight_range=None)



# 定义不同类型的代理人及其对应的类
AGENT_TYPES = {'F': FireFighter, 'B': Builder, 'G': Generalist}

# 定义地图的大小
MAP_SIZE = 16

# 定义火灾发生的最小概率
MIN_FIRE_START_RATE = 10
# 定义火灾发生的最大概率
MAX_FIRE_START_RATE = 80

# 定义不同类型的建筑物及其对应的类
BUILDING_TYPES = {'F': FastBurnBuilding, 'S': SlowBurnBuilding}
# 定义不同类型建筑物出现的概率
BUILDING_PROBS = {'F': 0.5, 'S': 0.5}

# 定义消防员在火灾环境中的移动速度
FF_F_SPEED = 1.0
# 定义建造者在火灾环境中的移动速度
FF_B_SPEED = 0.05
# 定义建造者在建筑环境中的移动速度
B_F_SPEED = 0.05
# 定义建造者在建筑环境中的移动速度
B_B_SPEED = 1.0
# 定义通用代理人在火灾环境中的唯一标识
F_ID = len(BUILDING_TYPES)
# 定义建造者在建筑环境中的唯一标识
B_ID = len(BUILDING_TYPES) + 1
# 定义通用代理人在建筑环境中的唯一标识
G_ID = len(BUILDING_TYPES) + 2



class FireFightersEnv(MultiAgentEnv):
    def __init__(self,
                 entity_scheme=True,     # 是否使用实体表示方案（强制要求为True）
                 heuristic_alloc=False,  # 是否使用启发式分配策略
                 scenario_dict=None,     # 场景配置字典（包含训练/测试场景参数）
                 episode_limit=150,      # 单次episode的最大时间步长
                 game_config=None,       # 游戏核心参数配置（火灾/建造速率等）
                 reward_config=None,     # 奖励函数配置参数
                 end_on_any_burn=False,  # 是否在任何建筑烧毁时结束episode
                 reward_scale=20,        # 奖励缩放因子
                 track_ac_type=False,    # 是否跟踪智能体类型
                 seed=0):                # 随机种子

        # 强制要求使用实体表示方案（entity_scheme必须为True）
        assert entity_scheme, "This environment only supports entity_scheme"

        self.heuristic_alloc = heuristic_alloc   # 存储启发式分配标志

        # 场景配置处理（分无限场景和预设场景两种模式）
        if scenario_dict == 'infinite':
            # 无限模式配置（无固定场景）
            self.train_scenarios = None
            self.test_scenarios = None
            self.bld_spacing = 3  # 建筑间最小间隔
            self.max_n_agents = 8  # 最大智能体数量
            self.max_n_buildings = 8  # 最大建筑数量
        else:
            # 预设场景配置（从字典加载参数）
            self.train_scenarios = scenario_dict['train_scenarios']  # 训练场景集
            self.test_scenarios = scenario_dict['test_scenarios']  # 测试场景集
            self.max_n_agents = scenario_dict['max_n_agents']  # 最大智能体数
            self.max_n_buildings = scenario_dict['max_n_buildings']  # 最大建筑数
            self.bld_spacing = scenario_dict['bld_spacing']  # 建筑间隔

        # 生成建筑位置候选网格（基于地图尺寸和建筑间隔）
        self.bld_pos_candidates = list(product(
            range(1, MAP_SIZE, self.bld_spacing),
            range(1, MAP_SIZE, self.bld_spacing)
        ))

        self.episode_limit = episode_limit      # 存储episode步数上限
        self.end_on_any_burn = end_on_any_burn  # 存储建筑烧毁结束标志

        # 游戏核心参数默认配置（火灾/建造相关参数）
        if game_config is None:
            self.game_config = {
                'init_fire_chance': 0.4,   # 建筑初始着火概率
                'fire_burn_rate': 0.035,   # 建筑燃烧速率（每秒损失生命值）
                'fire_reduce_rate': 0.25,  # 消防员灭火效率（每秒减少火势）
                'fire_growth_rate': 1.1,   # 火势自然增长率
                'base_build_rate': 0.125,  # 基础建造速率（每秒恢复生命值）
                'generalist_help_factor': 0.45  # 多面手协助效率因子
            }
        else:
            self.game_config = game_config  # 使用传入的配置

        # 奖励函数默认配置
        if reward_config is None:
            self.reward_config = {
                'health_delta_mult': 10.0,  # 建筑生命值变化乘数
                'complete': 1.0,            # 建筑完全修复奖励
                'burned_down': -20.0,       # 建筑烧毁惩罚
                'extinguish': 0.5,          # 灭火成功奖励
                'solved': 20.0,             # 任务完成奖励
                'ts_pen': 0.0               # 时间步惩罚（可能为负奖励）
            }
        else:
            self.reward_config = reward_config  # 使用传入的配置

        # 计算最大可能奖励（用于归一化）
        self.max_reward = 0.5 * self.reward_config['health_delta_mult'] * self.max_n_buildings
        self.max_reward += self.reward_config['complete'] * self.max_n_buildings
        self.max_reward += self.reward_config['solved']

        self.reward_scale = reward_scale    # 奖励缩放因子:20
        self.track_ac_type = track_ac_type  # 智能体类型跟踪标志:False

        # 计算动作空间维度（基于智能体类型）
        max_move_dist = max(a(0).max_move_dist for a in AGENT_TYPES.values())
        self.n_actions = 1 + (4 * max_move_dist) + 3  # 动作组成：
        # 1: 停留(stay)
        # 4 * max_move_dist: 四个方向移动（不同距离）
        # 3: 灭火(put out fire)/建造(build)/冻结(freeze)

        # 初始化渲染和状态跟踪变量
        self._render = None  # 渲染器占位符
        self.time = 0        # 当前时间步计数器
        self.seed(seed)      # 设置随机种子
        self.grid = np.array([[None for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)])  # 地图网格初始化
        self.agents = []     # 智能体列表初始化     智能体数量的获取！！
        self.buildings = []  # 建筑列表初始化
    @property
    def objects(self):
        return self.agents + self.buildings

    def _vis_objs(self, x, y, vis_range):
        gridcopy = self.grid.copy()
        min_x = max(0, x - vis_range)
        max_x = min(MAP_SIZE, x + vis_range + 1)
        min_y = max(0, y - vis_range)
        max_y = min(MAP_SIZE, y + vis_range + 1)
        local_region = gridcopy[min_x:max_x, min_y:max_y].flatten().tolist()
        return [obj for obj in local_region if obj is not None]

    def _get_dir_obj(self, x, y, direction, dist=1):
        if direction == 'N':
            if y + dist >= MAP_SIZE:
                return 'boundary'
            else:
                return self.grid[x, y + dist]
        elif direction == 'E':
            if x + dist >= MAP_SIZE:
                return 'boundary'
            else:
                return self.grid[x + dist, y]
        elif direction == 'S':
            if y - dist < 0:
                return 'boundary'
            else:
                return self.grid[x, y - dist]
        elif direction == 'W':
            if x - dist < 0:
                return 'boundary'
            else:
                return self.grid[x - dist, y]

    def _update_grid_pos(self):
        self.grid[:, :] = None
        for agent in self.agents:
            self.grid[agent.x, agent.y] = agent
        for building in self.buildings:
            self.grid[building.x, building.y] = building

    def seed(self, seed=None):
        np.random.seed(seed)

    def sample(self, test=False):
        if self.train_scenarios is None:
            n_agents = np.random.randint(2, self.max_n_agents + 1)
            n_buildings = np.random.randint(2, self.max_n_buildings + 1)
            agent_units = [np.random.choice(list(AGENT_TYPES.keys())) for _ in range(n_agents)]
            building_units = [np.random.choice(list(BUILDING_TYPES.keys())) for _ in range(n_buildings)]
            self.curr_scenario = (agent_units, building_units)
            return
        if test:
            ind = np.random.randint(len(self.test_scenarios))
            self.curr_scenario = self.test_scenarios[ind]
        else:
            ind = np.random.randint(len(self.train_scenarios))
            self.curr_scenario = self.train_scenarios[ind]

    def set_index(self, index, test=False):
        if test:
            self.curr_scenario = self.test_scenarios[index]
        else:
            self.curr_scenario = self.train_scenarios[index]

    def set_n_tasks(self, n_tasks, test=False):
        # TODO: do this more efficiently
        while True:
            self.sample(test=test)
            agent_units, (min_n_blds, max_n_blds) = self.curr_scenario
            if n_tasks >= min_n_blds and n_tasks <= max_n_blds:
                self.curr_scenario = (agent_units, (n_tasks, n_tasks))
                break

    def reset(self, index=None, test=False, n_tasks=None):
        # 重置环境状态的核心方法
        # index: 指定场景索引；test: 是否为测试模式；n_tasks: 指定任务数量

        # 场景选择逻辑
        if index is None and n_tasks is None:
            # 随机采样场景（训练或测试模式）
            self.sample(test=test)
        else:
            # 确保只指定一种选择方式
            assert not (index is not None and n_tasks is not None), "Can only specify one of index or n_tasks"
            if index is not None:
                # 按索引设置场景
                self.set_index(index, test=test)
            elif n_tasks is not None:
                # 按任务数量设置场景
                self.set_n_tasks(n_tasks, test=test)

        # 加载当前场景配置
        self.agent_units, building_spec = self.curr_scenario
        self.n_agents = len(self.agent_units)  # 获取智能体数量

        # 建筑生成逻辑
        if len(building_spec) == 2 and type(building_spec[0]) is int:
            # 随机生成建筑数量和类型
            min_n_blds, max_n_blds = building_spec
            self.n_buildings = np.random.randint(min_n_blds, max_n_blds + 1)
            b_types = sorted(BUILDING_TYPES.keys())
            b_probs = [BUILDING_PROBS[btyp] for btyp in b_types]
            self.building_units = [np.random.choice(b_types, p=b_probs)
                                   for _ in range(self.n_buildings)]
        else:
            # 使用预设建筑配置
            self.n_buildings = len(building_spec)
            self.building_units = building_spec

        # 重置环境状态记录
        self.agents = []  # 清空智能体列表
        self.buildings = []  # 清空建筑列表
        self.ep_info = {}  # 清空episode信息
        self.time = 0  # 重置时间步计数器

        # 创建智能体位置网格（防止与建筑重叠）
        agt_pos_grid = np.array([[None for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)])
        obj_id = 0  # 对象ID计数器

        # 标记建筑候选位置为不可占用
        bx, by = list(zip(*self.bld_pos_candidates))
        agt_pos_grid[bx, by] = ' '

        # 智能体位置优化：逐步缩小可用区域使智能体聚集[6,8](@ref)
        shrink_grid = agt_pos_grid.copy()
        idx = 0
        while (shrink_grid == None).sum() >= self.n_agents:
            agt_pos_grid = shrink_grid.copy()
            shrink_grid[idx, :] = ' '  # 收缩上边界
            shrink_grid[-idx - 1, :] = ' '  # 收缩下边界
            shrink_grid[:, idx] = ' '  # 收缩左边界
            shrink_grid[:, -idx - 1] = ' '  # 收缩右边界
            idx += 1

        # 获取最终可用的智能体位置
        agt_pos_candidates = list(zip(*np.where(agt_pos_grid == None)))

        # 初始化智能体[8](@ref)
        for au in self.agent_units:
            agent = AGENT_TYPES[au](obj_id=obj_id)  # 创建智能体对象
            # 随机分配位置并从候选池移除
            agent.x, agent.y = agt_pos_candidates.pop(np.random.randint(len(agt_pos_candidates)))
            self.agents.append(agent)
            obj_id += 1  # 更新对象ID

        # 初始化建筑
        bld_pos_candidates_rem = self.bld_pos_candidates.copy()
        for bu in self.building_units:
            # 随机选择建筑位置
            x, y = bld_pos_candidates_rem.pop(np.random.randint(len(bld_pos_candidates_rem)))

            # 随机设置建筑初始健康状态（两种模式）
            if np.random.rand() < 0.5:
                init_health_range = (0.25, 0.4)  # 低健康度模式
            else:
                init_health_range = (0.6, 0.75)  # 高健康度模式

            # 创建建筑对象
            building = BUILDING_TYPES[bu](x=x, y=y,
                                          obj_id=obj_id,
                                          init_health_range=init_health_range,
                                          game_config=self.game_config)
            self.buildings.append(building)
            obj_id += 1  # 更新对象ID

        # 随机设置建筑着火状态[1](@ref)
        for bld in self.buildings:
            if np.random.rand() < self.game_config['init_fire_chance']:
                bld.set_fire()  # 根据配置概率点燃建筑

        # 更新网格位置信息
        self._update_grid_pos()

        # 初始化任务完成状态
        self.tasks_complete = [0 for _ in range(self.n_buildings)] + [1 for _ in
                                                                      range(self.max_n_buildings - self.n_buildings)]

        # 返回初始状态信息[6](@ref)
        return self.get_entities(), self.get_masks()  # 实体状态和掩码矩阵

    def get_entities(self):
        """
        获取所有实体的表示。
        Args:
            无
        Returns:
            list: 包含所有实体表示的列表，每个实体表示为一个一维的numpy数组。
        """
        nf_entity = self.get_entity_size()
        all_entities = []
        avail_actions = self.get_avail_actions()
        for obj in self.objects:
            ind = 0
            # entity type
            curr_ent = np.zeros(nf_entity, dtype=np.float32)
            curr_ent[ind + obj.ent_id] = 1
            ind += len(BUILDING_TYPES) + len(AGENT_TYPES)
            # one-hot x-y loc
            curr_ent[ind + obj.x] = 1
            ind += MAP_SIZE
            curr_ent[ind + obj.y] = 1
            ind += MAP_SIZE
            # scalar x-y loc
            curr_ent[ind] = obj.x / MAP_SIZE
            curr_ent[ind + 1] = obj.y / MAP_SIZE
            ind += 2
            # avail actions
            if obj.ent_type == 'agent':
                for iac in range(self.n_actions):
                    curr_ent[ind + iac] = avail_actions[obj.obj_id][iac] >= 1
            ind += self.n_actions
            # building properties
            if obj.ent_type == 'building':
                curr_ent[ind] = obj.health
                curr_ent[ind + 1] = int(obj.complete)
                curr_ent[ind + 2] = int(obj.burned_down)
                curr_ent[ind + 3] = obj.fire_strength
                curr_ent[ind + 4] = int(obj.fire_strength > 0.0)
                curr_ent[ind + 5] = obj.time_since_last_fire / obj.my_fire_start_rate
                # curr_ent[ind + 6] = obj.my_fire_start_rate / MAX_FIRE_START_RATE
                curr_ent[ind + 6] = (1 / obj.my_fire_start_rate - 1 / MAX_FIRE_START_RATE) / ((1 / MIN_FIRE_START_RATE) - (1 / MAX_FIRE_START_RATE))
            all_entities.append(curr_ent)
            # pad entities to fixed number across episodes (for easier batch processing)
            if obj.obj_id == self.n_agents - 1:
                all_entities += [np.zeros(nf_entity, dtype=np.float32)
                                 for _ in range(self.max_n_agents -
                                                self.n_agents)]
            elif obj.obj_id == self.n_agents + self.n_buildings - 1:
                all_entities += [np.zeros(nf_entity, dtype=np.float32)
                                 for _ in range(self.max_n_buildings -
                                                self.n_buildings)]
        '''
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Entities: {len(all_entities)}")
        print(all_entities)
        print(f"Entities shape: {np.array(all_entities).shape}")
        print(f"Entities dtype: {np.array(all_entities).dtype}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        '''
        return all_entities

    def get_entity_size(self):
        nf_entity = 0
        # entity type
        nf_entity += len(AGENT_TYPES) + len(BUILDING_TYPES)
        # one-hot location coordinates
        nf_entity += 2 * MAP_SIZE
        # scalar location coordinates
        nf_entity += 2
        # available actions (only for agents)
        nf_entity += self.n_actions
        # building only properties (build completion, fire level)
        nf_entity += 7
        return nf_entity # 58

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.max_n_agents):
            if agent_id < self.n_agents:
                avail_agent = self.get_avail_agent_actions(agent_id)
            else:
                avail_agent = [1] + [0] * (self.n_actions - 1)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, a_id):
        agent = self.agents[a_id]
        # stay, move NESW, put out fire, build, freeze
        avail_actions = [0] * self.n_actions
        avail_actions[0] = 1

        for dist in range(1, agent.max_move_dist + 1):
            n_obj = self._get_dir_obj(agent.x, agent.y, 'N', dist=dist)
            e_obj = self._get_dir_obj(agent.x, agent.y, 'E', dist=dist)
            s_obj = self._get_dir_obj(agent.x, agent.y, 'S', dist=dist)
            w_obj = self._get_dir_obj(agent.x, agent.y, 'W', dist=dist)
            if dist == 1:
                surr_objs = [n_obj, e_obj, s_obj, w_obj]
            # can move into space if no building/agent occupies
            ind_start = 1 + (dist - 1) * 4
            avail_actions[ind_start] = int(n_obj is None)
            avail_actions[ind_start + 1] = int(e_obj is None)
            avail_actions[ind_start + 2] = int(s_obj is None)
            avail_actions[ind_start + 3] = int(w_obj is None)
        for obj in surr_objs:
            if obj is not None and obj != 'boundary' and obj.ent_type == 'building':
                bld_id = obj.obj_id - self.n_agents
                # can only put out fire if fire exists
                if obj.fire_strength > 0.0:
                    avail_actions[-3] = 2 + bld_id
                    # some agents can prevent fire from doing damage
                    if agent.can_freeze:
                        avail_actions[-1] = 2 + bld_id
                # can only build if building is not complete/burned down and not on fire
                if not (obj.burned_down or obj.complete) and obj.fire_strength == 0.0:
                    avail_actions[-2] = 2 + bld_id
                # only max of one building surrounding due to positioning constraints
                break
        return avail_actions

    def get_masks(self):
        """
        Returns:
        1) per entity observability mask over all entities (unoberserved = 1, else 0)
        2) mask of inactive entities (including enemies) over all possible entities
        3) entity to task matrix (agents are left unassigned unless we're using
            a task allocation heuristic)
        4) task mask vector (what tasks are available)
        """
        masks = {}
        obs_mask = np.ones((self.n_agents + self.n_buildings, self.n_agents + self.n_buildings),
                           dtype=np.uint8)
        for agent in self.agents:
            if agent.sight_range is not None:
                vis_objs = self._vis_objs(agent.x, agent.y, agent.sight_range)
            else:
                vis_objs = self.objects
            for obj in vis_objs:
                obs_mask[agent.obj_id, obj.obj_id] = 0
        obs_mask[self.n_agents:, self.n_agents:] = 0

        obs_mask_padded = np.ones((self.max_n_agents + self.max_n_buildings,
                                   self.max_n_agents + self.max_n_buildings),
                                  dtype=np.uint8)
        obs_mask_padded[:self.n_agents,
                        :self.n_agents] = obs_mask[:self.n_agents, :self.n_agents]
        obs_mask_padded[:self.n_agents,
                        self.max_n_agents:self.max_n_agents + self.n_buildings] = (
                            obs_mask[:self.n_agents, self.n_agents:]
        )
        obs_mask_padded[self.max_n_agents:self.max_n_agents + self.n_buildings,
                        :self.n_agents] = obs_mask[self.n_agents:, :self.n_agents]
        obs_mask_padded[self.max_n_agents:self.max_n_agents + self.n_buildings,
                        self.max_n_agents:self.max_n_agents + self.n_buildings] = (
                            obs_mask[self.n_agents:, self.n_agents:]
        )
        masks['obs_mask'] = obs_mask_padded

        entity_mask = np.ones(self.max_n_agents + self.max_n_buildings,
                              dtype=np.uint8)
        entity_mask[:self.n_agents] = 0
        entity_mask[self.max_n_agents:self.max_n_agents + self.n_buildings] = 0
        masks['entity_mask'] = entity_mask

        entity2task = np.ones((self.max_n_agents + self.max_n_buildings,
                               self.max_n_buildings), dtype=np.uint8)
        if self.heuristic_alloc:
            # get all active (i.e. not completed or burned down) buildings
            active_blds =[
                    bld for bld in self.buildings
                    if not (bld.complete or bld.burned_down)
                ]
            for ai, agent in enumerate(self.agents):
                # get manhattan distance from current agent to active buildings
                dist_blds = [
                    (abs(agent.x - bld.x) + abs(agent.y - bld.y), bld)
                    for bld in active_blds
                ]
                # get buildings sorted by closest first
                sorted_blds = [bld for _, bld in sorted(dist_blds, key=lambda x: x[0])]
                # get sorted buildings currently on fire
                fire_blds = [bld for bld in sorted_blds if bld.fire_strength > 0.0]
                # get sorted buildings not on fire and will not soon be on fire (must not be on fire in order to re-build)
                not_fire_blds = [bld for bld in sorted_blds if bld.fire_strength == 0.0]
                if agent.ent_id == F_ID:
                    # firefighter
                    priority_blds = fire_blds + not_fire_blds
                elif agent.ent_id == B_ID:
                    # builder
                    priority_blds = not_fire_blds + fire_blds
                elif agent.ent_id == G_ID:
                    # generalist (goes to fire since it can freeze those)
                    priority_blds = fire_blds + not_fire_blds
                # assign agent to closest prioritized (based on agent type) building
                if len(priority_blds) > 0:
                    # no need to assign if no buildings are available since
                    # episode is over in that case
                    assgn_bld_id = priority_blds[0].obj_id - self.n_agents
                    entity2task[ai, assgn_bld_id] = 0

        for bi in range(self.n_buildings):
            entity2task[self.max_n_agents + bi, bi] = 0

        masks['entity2task_mask'] = entity2task
        masks['task_mask'] = np.array(self.tasks_complete, dtype=np.uint8)

        return masks

    def step(self, actions):
        '''
            关键设计解析
            1.动作执行流程：
                移动动作：智能体可在NESW四个方向移动不同距离（max_move_dist控制最大步长）
                交互动作：灭火(F)、建造(B)、冻结(Z)三种建筑交互动作
                碰撞处理：采用曼哈顿距离检测，随机选择冲突智能体返回原位
            2.奖励计算系统：
                分层奖励：基础时间步惩罚 + 建筑健康变化奖励 + 灭火/建造事件奖励
                子任务奖励：为每个建筑单独计算奖励，支持课程学习
                终局奖励：完全解决额外奖励（solved配置项）
            3.终止条件判断：
                成功条件：所有建筑完成（complete）或烧毁（burned_down）
                失败条件：end_on_any_burn开启时任一建筑烧毁即终止
                超时条件：达到episode_limit时间步限制
            4.数据统计机制：
                动作跟踪：记录各类动作比例（需track_ac_type启用）
                任务状态：动态更新tasks_complete和tasks_changed
                性能指标：计算建筑完成比例(prop_buildings_completed)等关键指标
        '''
        # 将动作转换为整数列表，并截取有效智能体动作
        actions = [int(a) for a in actions[:self.n_agents]]

        # 验证每个智能体的动作是否可用
        for agent in self.agents:
            avail_actions = self.get_avail_agent_actions(agent.obj_id)
            assert avail_actions[actions[agent.obj_id]] >= 1, \
                "Agent {} cannot perform action {}".format(agent.obj_id,
                                                           actions[agent.obj_id])

        # 执行移动动作
        for agent in self.agents:
            action = actions[agent.obj_id]
            agent.prev_x = agent.x  # 记录移动前位置
            agent.prev_y = agent.y
            # 动作类型：0=停留，1~4*max_move_dist=移动，最后3个=建筑交互
            if action == 0:  # 停留动作
                agent.last_action = ' '

            # 处理移动动作（NESW四个方向）
            for move_dist in range(1, agent.max_move_dist + 1):
                ind_start = 1 + (move_dist - 1) * 4  # 计算动作索引起始位置
                if action == ind_start:  # 向北移动
                    agent.y += move_dist
                    agent.last_action = 'N%i' % (move_dist)
                elif action == ind_start + 1:  # 向东移动
                    agent.x += move_dist
                    agent.last_action = 'E%i' % (move_dist)
                elif action == ind_start + 2:  # 向南移动
                    agent.y -= move_dist
                    agent.last_action = 'S%i' % (move_dist)
                elif action == ind_start + 3:  # 向西移动
                    agent.x -= move_dist
                    agent.last_action = 'W%i' % (move_dist)

            # 处理建筑交互动作（灭火/建造/冻结）
            if action >= self.n_actions - 3:
                # 获取智能体当前位置的建筑对象
                act_bld = list(filter(lambda x: x.ent_type == 'building',
                                      self._vis_objs(agent.x, agent.y, 1)))[0]

                if action == self.n_actions - 3:  # 灭火动作
                    act_bld.reduce_fire(agent.fire_speed, agent.ent_id)
                    agent.fire_actions += 1  # 记录灭火次数
                    agent.last_action = 'F'
                elif action == self.n_actions - 2:  # 建造动作
                    act_bld.build(agent.build_speed, agent.ent_id)
                    agent.build_actions += 1  # 记录建造次数
                    agent.last_action = 'B'
                elif action == self.n_actions - 1:  # 冻结动作
                    act_bld.freeze()
                    agent.freeze_actions += 1  # 记录冻结次数
                    agent.last_action = 'Z'

        # 初始化奖励和子任务奖励
        total_reward = self.reward_config['ts_pen']  # 时间步惩罚
        subtask_rewards = np.zeros(self.max_n_buildings)  # 子任务奖励数组

        # 计算建筑状态变化和奖励
        for bi, building in enumerate(self.buildings):
            bld_info = building.step()  # 更新建筑状态
            bld_reward = 0
            # 计算建筑相关奖励
            bld_reward += self.reward_config['health_delta_mult'] * bld_info['buildings_health_delta']  # 健康值变化
            bld_reward += self.reward_config['burned_down'] * bld_info['buildings_burned']  # 建筑烧毁惩罚
            bld_reward += self.reward_config['extinguish'] * bld_info['fires_extinguished']  # 灭火奖励

            # 总奖励 = 建筑奖励 + 建筑完成奖励
            total_reward += bld_reward + self.reward_config['complete'] * bld_info['buildings_completed']
            # 子任务奖励 = 建筑奖励 + 任务完成奖励 + 时间步惩罚
            subtask_rewards[bi] += bld_reward + self.reward_config['solved'] * bld_info['buildings_completed']
            subtask_rewards[bi] += self.reward_config['ts_pen']

            # 记录episode信息
            for k, v in bld_info.items():
                self.ep_info[k] = self.ep_info.get(k, 0) + v

        # 处理智能体碰撞
        for ia, agent in enumerate(self.agents):
            for oa in range(ia + 1, self.n_agents):
                other = self.agents[oa]
                dist = abs(agent.x - other.x) + abs(agent.y - other.y)  # 曼哈顿距离
                if dist == 0:  # 发生碰撞
                    # 随机选择一个智能体返回原位置
                    chosen_agent = np.random.choice([agent, other])
                    (chosen_agent.x, chosen_agent.y) = (chosen_agent.prev_x,
                                                        chosen_agent.prev_y)

        self._update_grid_pos()  # 更新网格位置
        self.time += 1  # 增加时间步

        # 准备返回信息
        info = {}

        # 更新任务完成状态
        new_tasks_complete = [int(b.complete or b.burned_down) for b in self.buildings] + [1 for _ in range(
            self.max_n_buildings - self.n_buildings)]
        tasks_changed = any(last_comp != curr_comp for last_comp, curr_comp
                            in zip(self.tasks_complete, new_tasks_complete))
        self.tasks_complete = new_tasks_complete

        # 添加任务相关信息
        info["task_rewards"] = (subtask_rewards * self.reward_scale) / self.max_reward  # 缩放后的任务奖励
        info["tasks_terminated"] = self.tasks_complete  # 任务终止状态
        info["tasks_changed"] = tasks_changed  # 任务状态变化标志

        # 判断episode是否结束
        done = False
        # 条件1：任何建筑烧毁或所有建筑完成
        if self.end_on_any_burn and (
                all(b.complete for b in self.buildings) or any(b.burned_down for b in self.buildings)):
            done = True
        # 条件2：所有建筑完成或烧毁
        elif all((b.complete or b.burned_down) for b in self.buildings):
            done = True
        # 条件3：达到episode时间限制
        elif self.time >= self.episode_limit:
            done = True
            info["episode_limit"] = True  # 标记因时间限制结束

        # 如果episode结束，计算最终统计信息
        if done:
            info["prop_buildings_completed"] = float(sum(b.complete for b in self.buildings)) / float(len(self.buildings))  # 完成比例
            info["solved"] = int(all(b.complete for b in self.buildings))  # 是否完全解决
            if info["solved"] == 1:  # 完全解决奖励
                total_reward += self.reward_config['solved']

            # 计算平均统计信息
            for k, v in self.ep_info.items():
                if "-count" in k:  # 跳过计数器
                    continue
                if f"{k}-count" in self.ep_info.keys():  # 计算平均值
                    info[k] = float(v) / float(self.ep_info[f"{k}-count"])
                else:
                    info[k] = float(v) if isinstance(v, (int, float)) else 0.0

            # 如果启用动作类型跟踪
            if self.track_ac_type:
                ac_types = np.zeros((len(AGENT_TYPES), 3))  # 初始化动作类型矩阵
                for agent in self.agents:
                    # 记录各类动作次数
                    ac_types[agent.ent_id - len(BUILDING_TYPES), 0] += agent.fire_actions
                    ac_types[agent.ent_id - len(BUILDING_TYPES), 1] += agent.build_actions
                    ac_types[agent.ent_id - len(BUILDING_TYPES), 2] += agent.freeze_actions
                total_acs = ac_types.sum(axis=1, keepdims=True)  # 计算总动作
                total_acs[total_acs == 0] = 1  # 避免除零
                ac_types = ac_types / total_acs  # 计算动作比例
                # 只返回均值，不返回整个数组
                info['action_types_mean'] = float(np.mean(ac_types))

        # 缩放总奖励
        total_reward = (total_reward * self.reward_scale) / self.max_reward

        return total_reward, done, info  # 返回奖励、结束标志和信息

    def get_stats(self):
        return {}
    
    #-------------------------------------------------------------------------------------------
    #---------------------------------------------新加的拼接所有的实体特征的代码(成功运行)-----------------------------------------
    #-------------------------------------------------------------------------------------------
    '''
    def get_obs(self):
        entities = self.get_entities()
        masks = self.get_masks()
        obs_mask = masks['obs_mask']
        obs = []
        for agent_id in range(self.max_n_agents):
            if agent_id < self.n_agents:
                agent_obs = []
                for ent_id, ent in enumerate(entities):
                    if obs_mask[agent_id][ent_id] == 0:
                        agent_obs.append(ent)
                    else:
                        agent_obs.append(np.zeros_like(ent))
                obs.append(np.stack(agent_obs))  # shape: (n_entities, entity_feature)
            else:
                obs.append(np.zeros((self.max_n_agents + self.max_n_buildings,
                                    self.get_entity_size()), dtype=np.float32))
        return obs


    def get_obs_size(self):
        return self.get_entity_size() * (self.max_n_agents + self.max_n_buildings)

    def get_state(self):
        obs = self.get_obs()  # List of (n_entities, entity_feat)
        padded_obs = []
        for agent_id in range(self.max_n_agents):
            if agent_id < self.n_agents:
                padded_obs.append(obs[agent_id])
            else:
                padded_obs.append(np.zeros_like(obs[0]))
        return np.concatenate(padded_obs, axis=0).flatten()


    def get_state_size(self):
        return self.max_n_agents * (self.max_n_agents + self.max_n_buildings) * self.get_entity_size()
    '''
    #-------------------------------------------------------------------------------------------
    #---------------------------------------------新加的拼接所有的实体特征的代码(成功运行)-----------------------------------------
    #-------------------------------------------------------------------------------------------


    #-------------------------------------------------------------------------------------------
    #---------------------------------------------新加的固定obs维度在80左右的代码-----------------------------------------
    #-------------------------------------------------------------------------------------------
    def get_obs(self):
        obs = []

        for agent_id in range(self.max_n_agents):
            if agent_id < self.n_agents:
                agent = self.agents[agent_id]
                agent_obs = []

                # 1. 自身信息（7维）
                agent_obs.append(agent.ent_id / len(AGENT_TYPES))   # 类型归一化
                agent_obs.append(agent.x / MAP_SIZE)
                agent_obs.append(agent.y / MAP_SIZE)
                agent_obs.append(agent.fire_speed)
                agent_obs.append(agent.build_speed)
                agent_obs.append(float(agent.can_freeze))
                agent_obs.append(agent.max_move_dist / MAP_SIZE)

                # 2. 其他 agent（每个3维）
                for i in range(self.max_n_agents):
                    if i != agent_id:
                        if i < self.n_agents:
                            other = self.agents[i]
                            agent_obs.append(other.ent_id / len(AGENT_TYPES))
                            agent_obs.append(other.x / MAP_SIZE)
                            agent_obs.append(other.y / MAP_SIZE)
                        else:
                            agent_obs.extend([0.0, 0.0, 0.0])  # padding

                # 3. building 信息（每个6维）
                for i in range(self.max_n_buildings):
                    if i < self.n_buildings:
                        building = self.buildings[i]
                        agent_obs.append(building.x / MAP_SIZE)
                        agent_obs.append(building.y / MAP_SIZE)
                        agent_obs.append(building.health)
                        agent_obs.append(building.fire_strength)
                        agent_obs.append(float(building.complete))
                        agent_obs.append(float(building.burned_down))
                    else:
                        agent_obs.extend([0.0] * 6)

                # 断言验证
                assert len(agent_obs) == self.get_obs_size(), \
                    f"Agent {agent_id} obs dim mismatch: {len(agent_obs)} != {self.get_obs_size()}"

                obs.append(np.array(agent_obs, dtype=np.float32))

            else:
                # 无效 agent padding
                obs.append(np.zeros(self.get_obs_size(), dtype=np.float32))

        return obs


    def get_obs_size(self):
        n_self = 7
        n_other_agents = (self.max_n_agents - 1) * 3
        n_buildings = self.max_n_buildings * 6
        return n_self + n_other_agents + n_buildings

    def get_state(self):
        obs = self.get_obs()
        state = np.concatenate(obs, axis=0)
        assert state.shape[0] == self.get_state_size(), \
            f"State shape mismatch: got {state.shape[0]}, expected {self.get_state_size()}"
        return state

    def get_state_size(self):
        return self.get_obs_size() * self.max_n_agents

    #-------------------------------------------------------------------------------------------
    #---------------------------------------------新加的固定obs维度在80左右的代码-----------------------------------------
    #-------------------------------------------------------------------------------------------

    def get_env_info(self, args=None):
        env_info = {
            "entity_shape": self.get_entity_size(),
            "n_actions": self.n_actions,
            "n_agents": self.max_n_agents,
            "n_entities": self.max_n_agents + self.max_n_buildings,
            "episode_limit": self.episode_limit,
            "n_tasks": self.max_n_buildings,
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size()
        }
        # 只返回 int/float 类型，过滤掉 list 类型
        env_info = {k: v for k, v in env_info.items() if isinstance(v, (int, float))}
        return env_info

    def render(self, allocs=None, mode='human', close=False, verbose=True):
        self.init_render()
        self._render.new_frame()

        # color reference: https://matplotlib.org/stable/gallery/color/named_colors.html
        agent_type_colors = ['lightcoral', 'cornflowerblue', 'mediumseagreen', 'yellow']
        build_type_colors = ['dimgrey', 'black']
        build_type_background = ['whitesmoke', 'darkgrey']
        build_type_health = ['darkgrey', 'black']
        build_type_text = ['black', 'white']
        build_type_burned = ['crimson', 'firebrick']
        build_type_completed = ['goldenrod', 'goldenrod']

        action_dict = {'F': 'aqua', 'B': 'lime', 'Z': 'tomato'}

        bg = plt.Rectangle((0, 0), MAP_SIZE + 1, MAP_SIZE + 1,
                           linewidth=5, edgecolor='black', facecolor='white',
                           fill=True, zorder=1.3, alpha=1.0)
        self._render.add_artist(bg)
        for ia, agent in enumerate(self.agents):
            if agent.sight_range is not None:
                agent_view = plt.Rectangle((agent.x - agent.sight_range, agent.y - agent.sight_range), agent.sight_range * 2 + 1, agent.sight_range * 2 + 1,
                                           linewidth=5, edgecolor='black', facecolor='black', fill=True,
                                           zorder=1.4, alpha=0.1)
                self._render.add_artist(agent_view)

            agent_sq = plt.Circle((agent.x+0.5, agent.y+0.5), radius=0.4,
                                   facecolor=agent_type_colors[agent.ent_id - len(BUILDING_TYPES)],
                                   edgecolor='black',
                                   linewidth=4.0,
                                   zorder=1.5, alpha=1.0)
            self._render.add_artist(agent_sq)

            if agent.last_action in action_dict:
                agent_action = plt.Circle((agent.x+0.5, agent.y+0.5), radius=0.4,
                                             linewidth=6.0, edgecolor=action_dict[agent.last_action],
                                             fill=False,
                                             zorder=1.6, alpha=1.0)
                self._render.add_artist(agent_action)
            if allocs is not None:
                agent_desc = plt.Annotation(f"{allocs[ia]}",
                                            (agent.x+0.3, agent.y+0.3),
                                            fontsize=30.0)
                self._render.add_artist(agent_desc)

        for ib, bld in enumerate(self.buildings):
            build_sq = plt.Rectangle((bld.x, bld.y), 1, 1,
                                     facecolor=build_type_background[bld.ent_id],
                                     zorder=1.4, alpha=1.0)
            self._render.add_artist(build_sq)
            build_frame = plt.Rectangle((bld.x, bld.y), 1, 1,
                                     edgecolor=build_type_colors[bld.ent_id],
                                     linewidth=5, fill=None,
                                     zorder=1.6, alpha=1.0)
            self._render.add_artist(build_frame)
            if bld.complete:
                build_comp = plt.Rectangle((bld.x, bld.y), 1, 1,
                                           facecolor=build_type_completed[bld.ent_id], linewidth=5,
                                           zorder=1.55, alpha=1.0)
                self._render.add_artist(build_comp)
            if bld.burned_down:
                pass
            if bld.health > 0.0:
                build_health = plt.Rectangle((bld.x, bld.y), 0.5, bld.health,
                                             facecolor=build_type_health[bld.ent_id],
                                             zorder=1.5, alpha=1.0)
                self._render.add_artist(build_health)
            if bld.fire_strength > 0.0:
                fire_stat = plt.Rectangle((bld.x + 0.5, bld.y), 0.5, bld.fire_strength,
                                          facecolor='orangered',
                                          zorder=1.5, alpha=1.0)
                self._render.add_artist(fire_stat)

            annot_color = build_type_burned[bld.ent_id] if bld.burned_down else build_type_text[bld.ent_id]
            annot_pos = 0.3 if (bld.burned_down or bld.complete) else 0.41
            build_desc = plt.Annotation(f"{ib}",
                                        (bld.x + 0.3, bld.y + annot_pos),
                                        fontsize=30,
                                        fontweight='bold',
                                        color=annot_color)
            self._render.add_artist(build_desc)
            if not (bld.complete or bld.burned_down):
                build_next = plt.Annotation(f"{bld.time_to_next_fire}",
                                            (bld.x + 0.53, bld.y + 0.08),
                                            fontsize=13,
                                            fontweight='bold',
                                            color=build_type_text[bld.ent_id])
                self._render.add_artist(build_next)

        # self._render.fig.clear()
        image = self._render.draw()
        self._render.render(image)
        return image

    def init_render(self):
        if self._render is None:
            self._render = Render()
        return self
