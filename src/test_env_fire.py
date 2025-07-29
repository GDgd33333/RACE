from envs import REGISTRY

env = REGISTRY["firefighters"]()
print("环境创建成功！")
obs = env.reset()
print("初始观测:", obs.shape if hasattr(obs, 'shape') else obs)
action = [0] * len(env.agents)  # 所有agent执行"停留"
reward, done, info = env.step(action)
print("第一步结果 - 奖励:", reward, "终止:", done)