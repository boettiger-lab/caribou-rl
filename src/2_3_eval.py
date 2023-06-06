
# Initialize saved copy of eval environment:
config = agent.evaluation_config.env_config
config.update({'seed': 42})
env = agent.env_creator(config)

env.training = False
df = []
for rep in range(50):
  episode_reward = 0
  observation, _ = env.reset()
  for t in range(env.Tmax):
    action = agent.compute_single_action(observation)
    df.append(np.append([t, rep, episode_reward, action[0], action[1]], observation))
    observation, reward, terminated, done, info = env.step(action)
    episode_reward += reward
    if terminated:
      break
    
cols = ["t", "rep", "reward", "A1", "A2",  "X", "Y", "Z"]
df = pd.DataFrame(df, columns = cols)
df.to_csv(f"data/PPO{iterations}.csv.xz", index = False)



## Plots ## 
import plotnine
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path
## Timeseries
df = pd.read_csv(f"data/PPO{iterations}.csv.xz")
df2 = (df[df.rep == 3.0]
       .melt(id_vars=["t",  "reward", "rep"])
       .groupby(['t', "variable"], as_index=False)
       .agg({'reward': 'mean',
             'value': 'mean',
             #'action': 'mean'
             })) 
ggplot(df2, aes("t", "value", color="variable")) + geom_line()

## summary stats
reward = df[df.t == max(df.t)].reward
reward.mean()
np.sqrt(reward.var())

## quick policy plot
policy_df = []
states = np.linspace(-1,0.5,101)
for rep in range(10):
  obs, _ = env.reset()
  #obs[2] += .05 * rep
  for state in states:
      obs[0] = state
      action = agent.compute_single_action(obs)
      escapement = max(state + 1 - action[0], 0)
      policy_df.append([state+1, escapement, action[0], rep])
      
policy_df = pd.DataFrame(policy_df, columns=["observation","escapement","action","rep"])
ggplot(policy_df, aes("observation", "escapement", color = "rep")) + geom_point(shape=".")

