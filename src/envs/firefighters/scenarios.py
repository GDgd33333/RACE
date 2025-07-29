import numpy as np
from functools import partial
from itertools import combinations_with_replacement
from numpy.random import RandomState
from .firefighters import AGENT_TYPES, BUILDING_TYPES


def valid_build_scen(bld_units):
    """
    判断建筑物布局是否合理。

    Args:
        bld_units (list): 包含建筑物信息的列表，每个元素是一个字符串，表示一个建筑物的信息，格式为 "id,x,y,z"，其中 x, y, z 分别表示建筑物的位置。

    Returns:
        bool: 如果所有建筑物位置都不重复，则返回 True，否则返回 False。

    Notes:
        确保没有建筑物占用相同的位置。

    """
    # 确保没有建筑物占用相同的位置
    # ensures no buildings take up the same loc
    all_locs = set()
    for unit in bld_units:
        # 提取建筑物位置信息
        # loc_str = ','.join(unit.split(',')[1:])
        loc_str = ','.join(unit.split(',')[1:])
        all_locs.add(loc_str)
    return len(all_locs) == len(bld_units)



def get_all_unique_teams(all_types, min_len, max_len):
    """
    获取所有独特的团队组合。

    Args:
        all_types (list): 所有可能的团队成员类型列表。
        min_len (int): 组合的最小长度。
        max_len (int): 组合的最大长度。

    Returns:
        list: 包含所有独特的团队组合列表。

    """
    all_uniq = []
    # 循环遍历从min_len到max_len的每一个长度
    for i in range(min_len, max_len + 1):
        # 使用combinations_with_replacement函数生成长度为i的所有可能组合，并将结果添加到all_uniq列表中
        all_uniq += list(combinations_with_replacement(all_types, i))
    return all_uniq



def generate_scenarios(min_n_agents, max_n_agents, rs, max_diversity=True):
    """
    生成场景。
    Args:
        min_n_agents (int): 每个场景中的最小代理数。
        max_n_agents (int): 每个场景中的最大代理数。
        rs (RandomState): 随机数生成器。
        max_diversity (bool): 是否要求最大多样性。

    Returns:
        list: 包含代理团队和场景信息的列表。

    """
    # 我们按唯一代理团队拆分场景
    # (环境将生成所有可能的建筑配置)
    # we split scenarios by unique sets of agents
    # (environment will generate all possible building configurations)
    uniq_agent_teams = get_all_unique_teams(sorted(AGENT_TYPES.keys()), min_n_agents, max_n_agents)
    rs.shuffle(uniq_agent_teams)


    n_ag_types = len(AGENT_TYPES.keys())
    scenario_list = []
    for i in range(len(uniq_agent_teams)):
        # 获取当前唯一代理团队
        agent_scenario = list(uniq_agent_teams[i])
        n_agents = len(agent_scenario)
        scen_n_types = len(set(agent_scenario))
        # 如果代理类型数少于代理总数和代理类型总数的最小值，并且要求最大多样性，则跳过当前循环
        if scen_n_types < min(n_ag_types, n_agents) and max_diversity:
            continue


        # 将当前代理团队添加到场景列表中
        scenario_list.append((agent_scenario, (n_agents + 1, n_agents + 1)))
    return scenario_list



def scen_similarity(scen1, scen2):
    """
    计算两个场景之间的重叠系数。

    Args:
        scen1 (str): 第一个场景，类型为字符串。
        scen2 (str): 第二个场景，类型为字符串。

    Returns:
        float: 两个场景之间的重叠系数，范围在0到1之间。

    """
    """
    Return overlap coefficient between two scenarios
    """
    scen1_charcount = {}
    scen2_charcount = {}
    # 统计场景1中每个字符出现的次数
    for char1 in scen1:
        if char1 not in scen1_charcount:
            scen1_charcount[char1] = 1
        else:
            scen1_charcount[char1] += 1
    # 统计场景2中每个字符出现的次数
    for char2 in scen2:
        if char2 not in scen2_charcount:
            scen2_charcount[char2] = 1
        else:
            scen2_charcount[char2] += 1


    intersect = 0
    # union = 0
    # 计算两个场景中字符的交集数量
    for char in set(list(scen1_charcount.keys()) + list(scen2_charcount.keys())):
        intersect += min(scen1_charcount.get(char, 0), scen2_charcount.get(char, 0))
        # union += max(scen1_charcount.get(char, 0), scen2_charcount.get(char, 0))
    # return intersect / union
    # 返回两个场景中字符交集数量与较短场景长度的比值
    return intersect / min(len(scen1), len(scen2))



def rank_mean_similarity(compare_scenarios, rank_scenarios):
    """
    计算rank_scenarios中每个场景与compare_scenarios中场景的平均相似度，并返回按照平均相似度排序后的rank_scenarios的索引。

    Args:
        compare_scenarios (list of tuple): 待比较的场景列表，每个场景由若干字符组成。
        rank_scenarios (list of tuple): 待排序的场景列表，每个场景由若干字符组成。

    Returns:
        numpy.ndarray: 按照平均相似度排序后的rank_scenarios的索引。

    """
    """
    Return indices of rank_scenarios in order of mean similarity to
    compare_scenarios
    """
    # 将compare_scenarios中的场景转换为字符串列表
    comp_scen_strs = [''.join(scen[0]) for scen in compare_scenarios]
    # 将rank_scenarios中的场景转换为字符串列表
    rank_scen_strs = [''.join(scen[0]) for scen in rank_scenarios]


    # 计算每个rank_scenarios场景与compare_scenarios场景的平均相似度
    comp_scen_sims = [np.mean([scen_similarity(rank_sc, comp_sc)
                               # 对每个rank_scenarios场景，计算其与每个compare_scenarios场景的相似度
                               for comp_sc in comp_scen_strs])
                      # 对每个rank_scenarios场景，计算其与compare_scenarios场景的平均相似度
                      for rank_sc in rank_scen_strs]


    # 返回按平均相似度排序后的rank_scenarios索引
    return np.argsort(comp_scen_sims)



def generate_scen_dict(min_n_agents=2, max_n_agents=8, test_ratio=0.15,
                       train_ratio=0.25, bld_spacing=7, rs=None):
    """
    生成场景字典。

    Args:
        min_n_agents (int): 场景中最少智能体数量，默认为2。
        max_n_agents (int): 场景中最多智能体数量，默认为8。
        test_ratio (float): 测试场景占总场景的比例，默认为0.15。
        train_ratio (float): 训练场景占总场景的比例，默认为0.25。
        bld_spacing (int): 建筑物之间的最小间距，默认为7。
        rs (RandomState, optional): 随机数生成器，默认为None。

    Returns:
        dict: 包含训练场景、测试场景、最多智能体数量、最多建筑物数量以及建筑物间距的场景字典。

    Raises:
        AssertionError: 如果建筑物间距小于3，则抛出异常。

    """
    # 断言建筑物间距至少为3
    assert bld_spacing >= 3, "Buildings must be at least 3 spaces apart"
    if rs is None:
        rs = RandomState()


    # 生成所有场景
    all_scenarios = generate_scenarios(min_n_agents, max_n_agents, rs)
    # 测试场景在比例相同时总是固定的
    # test scenarios are always fixed if ratio is same
    n_test = int(test_ratio * len(all_scenarios))
    test_scenarios = all_scenarios[:n_test]
    if train_ratio is None:
        train_scenarios = all_scenarios
    else:
        n_train = int(train_ratio * len(all_scenarios))
        train_scenarios = all_scenarios[n_test:n_test + n_train]
    # 构建场景字典
    scenario_dict = {'train_scenarios': train_scenarios,
                     'test_scenarios': test_scenarios,
                     'max_n_agents': max_n_agents,
                     'max_n_buildings': max_n_agents + 1,
                     'bld_spacing': bld_spacing}
    return scenario_dict



def generate_scen_dict_sim(min_n_agents=2, max_n_agents=8, test_ratio=0.15,
                           train_pct_range=(0.0, 0.25), bld_spacing=7, rs=None):
    """
    生成场景字典，该字典包含训练场景、测试场景、最大代理数、最大建筑物数以及建筑物间距。

    Args:
        min_n_agents (int): 场景中的最小代理数，默认为2。
        max_n_agents (int): 场景中的最大代理数，默认为8。
        test_ratio (float): 测试场景占所有场景的比例，默认为0.15。
        train_pct_range (tuple): 训练场景占剩余场景的比例范围，默认为(0.0, 0.25)。
        bld_spacing (int): 建筑物之间的最小间距，默认为7。
        rs (RandomState, optional): 随机数生成器，默认为None。如果为None，则创建一个新的RandomState实例。

    Returns:
        dict: 包含训练场景、测试场景、最大代理数、最大建筑物数以及建筑物间距的场景字典。

    Raises:
        AssertionError: 如果建筑物间距小于3，则引发此异常。

    """
    # 断言建筑物之间的最小间距应至少为3
    assert bld_spacing >= 3, "Buildings must be at least 3 spaces apart"
    if rs is None:
        rs = RandomState()


    # 生成所有场景
    all_scenarios = generate_scenarios(min_n_agents, max_n_agents, rs)
    # 测试场景始终固定，如果比例相同
    # test scenarios are always fixed if ratio is same
    n_test = int(test_ratio * len(all_scenarios))
    test_scenarios = all_scenarios[:n_test]
    other_scenarios = all_scenarios[n_test:]


    # 计算其他场景与测试场景的平均相似度排名
    other_incr_sim_inds = rank_mean_similarity(test_scenarios, other_scenarios)
    min_rank_ind = int(train_pct_range[0] * len(other_scenarios))
    max_rank_ind = int(train_pct_range[1] * len(other_scenarios))
    # 获取训练场景索引
    train_scen_inds = other_incr_sim_inds[min_rank_ind:max_rank_ind]
    train_scenarios = [other_scenarios[i] for i in train_scen_inds]


    # 创建场景字典
    scenario_dict = {'train_scenarios': train_scenarios,
                     'test_scenarios': test_scenarios,
                     'max_n_agents': max_n_agents,
                     'max_n_buildings': max_n_agents + 1,
                     'bld_spacing': bld_spacing}
    return scenario_dict



def generate_single_scen_dict(agent_list=['F', 'F', 'F', 'B', 'B', 'B', 'G', 'G'],
                              building_list=['F', 'F', 'F', 'F', 'S', 'S', 'S', 'S'],
                              bld_spacing=7, rs=None):
    """
    生成包含单个场景的字典。

    Args:
        agent_list (list, optional): 代理列表，默认为['F', 'F', 'F', 'B', 'B', 'B', 'G', 'G']。
        building_list (list, optional): 建筑列表，默认为['F', 'F', 'F', 'F', 'S', 'S', 'S', 'S']。
        bld_spacing (int, optional): 建筑间距，默认为7。
        rs (object, optional): 随机种子生成器，默认为None。

    Returns:
        dict: 包含单个场景的字典。

    """
    """
    生成包含单个场景的字典。

    Args:
        agent_list (list, optional): 代理列表，默认为['F', 'F', 'F', 'B', 'B', 'B', 'G', 'G']。
        building_list (list, optional): 建筑列表，默认为['F', 'F', 'F', 'F', 'S', 'S', 'S', 'S']。
        bld_spacing (int, optional): 建筑间距，默认为7。
        rs (object, optional): 随机种子生成器，默认为None。

    Returns:
        dict: 包含单个场景的字典。

    """
    # 创建一个包含单个场景的列表
    all_scenarios = [(agent_list, building_list)]

    # 创建一个字典，用于存储场景信息
    scenario_dict = {'train_scenarios': all_scenarios,
                     'test_scenarios': all_scenarios,
                     'max_n_agents': len(agent_list),  # 最大代理数量
                     'max_n_buildings': len(building_list),  # 最大建筑数量
                     'bld_spacing': bld_spacing}  # 建筑间距

    # 返回场景字典
    return scenario_dict


'''
    2-5a_2-5b:
        1.使用 generate_scen_dict 函数生成。
        2.智能体数量范围为 2 到 5。
        3.可能包含两种变体（a 和 b），对应不同的场景布局或难度。


    6-10a_6-10b_sim_Q2:
        1.使用 generate_scen_dict_sim 函数生成
        2.智能体数量范围为 6 到 10
        3.包含两种变体（a 和 b）
        4.测试场景占总场景的 25%（Q2）(15%（Q1）)
        5.训练场景可能从剩余场景中按相似度排名的前 25% 选取
'''
scenarios = {
    '2-5a_2-5b': partial(generate_scen_dict,
                         min_n_agents=2, max_n_agents=5,
                         test_ratio=1.0, train_ratio=None,
                         bld_spacing=3),
    '2-8a_2-8b': partial(generate_scen_dict,
                         min_n_agents=2, max_n_agents=8,
                         test_ratio=1.0, train_ratio=None,
                         bld_spacing=3),
    '2-10a_2-10b': partial(generate_scen_dict,
                           min_n_agents=2, max_n_agents=10,
                           test_ratio=1.0, train_ratio=None,
                           bld_spacing=3),
    '12a_12b': partial(generate_scen_dict,
                       min_n_agents=12, max_n_agents=12,
                       test_ratio=1.0, train_ratio=None,
                       bld_spacing=3),
    '16a_16b': partial(generate_scen_dict,
                       min_n_agents=16, max_n_agents=16,
                       test_ratio=1.0, train_ratio=None,
                       bld_spacing=3),
    '20a_20b': partial(generate_scen_dict,
                       min_n_agents=20, max_n_agents=20,
                       test_ratio=1.0, train_ratio=None,
                       bld_spacing=3),
    '24a_24b': partial(generate_scen_dict,
                       min_n_agents=24, max_n_agents=24,
                       test_ratio=1.0, train_ratio=None,
                       bld_spacing=3),
    '2-8a_2-8b_sim_Q1': partial(generate_scen_dict_sim,
                                min_n_agents=2, max_n_agents=8,
                                test_ratio=0.15, train_pct_range=(0.0, 0.25)),
    '2-8a_2-8b_sim_Q2': partial(generate_scen_dict_sim,
                                min_n_agents=2, max_n_agents=8,
                                test_ratio=0.15, train_pct_range=(0.25, 0.5)),
    '2-8a_2-8b_sim_Q3': partial(generate_scen_dict_sim,
                                min_n_agents=2, max_n_agents=8,
                                test_ratio=0.15, train_pct_range=(0.5, 0.75)),
    '2-8a_2-8b_sim_Q4': partial(generate_scen_dict_sim,
                                min_n_agents=2, max_n_agents=8,
                                test_ratio=0.15, train_pct_range=(0.75, 1.0)),
    'single_scen': partial(generate_single_scen_dict,
                           agent_list=['F', 'F', 'F', 'B', 'B', 'B', 'G', 'G'],
                           building_list=['F', 'F', 'F', 'F', 'S', 'S', 'S', 'S'])
}
