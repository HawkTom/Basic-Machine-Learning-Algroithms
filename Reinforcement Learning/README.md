## Reinforcement Learning Report

### Pre Lab

**Introduction of reinforcement learning**

Reinforcement learning (RL) is an area of machine learning inspired by behaviourist psychology, concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. The image below is a very clear explanation of reinforcement learning.
<div align="center" style="zoom:70%"><img src = "https://d3ansictanv2wj.cloudfront.net/image3-5f8cbb1fb6fb9132fef76b13b8687bfc.png">
</div>

Reinforcement learning exits many algorithm since it was been proposed. In machine learning, the environment is typically formulated as a Markov decision process (MDP), as many reinforcement learning algorithms for this context utilize dynamic programming techniques. 

Basic reinforcement learning is modelled as Markov decision process (MDP)

1. a set of environment and agent states, S;
2. a set of actions, A, of the agent;
3. ${\displaystyle P_{a}(s,s')=Pr(s_{t+1}=s'|s_{t}=s,a_{t}=a)}$ is the probability of transition from state s to state s' under action a.
4. ${\displaystyle R_{a}(s,s')} R_a(s,s')$ is the immediate reward after transition from s to ${\displaystyle s'} s'$ with action a.
5. rules that describe what the agent observes

Based on MDP, some reinforcement learning method was been proposed. And it can be divided into two classes: model-based RL and model-free RL. In the real problem, model-free method is used more frequently the model-based RL. Some common model-free algorithms are Temporal Difference and other versions like Sarsa algorithm, Q-Learning algorithm and etc. With the Development of neural network,the Deep-Q-Network is been proposed based on Q-learning algorithm. The details of the algorithm will be introduced in next section.

**Openai gym environment**

Openai is a foundation of artificial general intelligence and it develop a software platforms named gym for reinforcement learning. 

Gym is a collection of environments/problems designed for testing and developing reinforcement learning algorithms—it saves the user from having to create complicated environments. 

The installation of gym platforms could refer to github documents.

<div style="page-break-before: always;"> </div>

### Algorithm - Q-Learning

Q -learning algorithm is a very common but efficient algorithm in reinforcement learning. 
<div align = "center" style="zoom:70%"><img src = "https://www.researchgate.net/profile/Kao-Shing_Hwang/publication/220776448/figure/fig1/AS:394068661161984@1470964698231/Fig-2-The-Q-Learning-Algorithm-6.png">
</div>

The pseudo code of Q-learning is showed above. It exits some important elements in the algorithm.

- $Q(s, a)$ is a action value function which is measured the quality of an action for corresponding states. It is a policy of decision process. The main intent of reinforcement learning is to get the optimal policy for a specific dynamic process.
- $Q(s, a)$ is a policy, $s$ represent all the states of process. $a$ represent all of the actions for all states. 
    - If the states and the action is not infinity and also discrete, the policy could be a table. We named it Q-table. 
    - If the states is continues or the states is infinity, we can not get the Q-table. An alternative way is network. The input is state, and the output is action. 
- Another important key is the update strategy. It shows at below.
  <img src ="https://d3ansictanv2wj.cloudfront.net/image2-edf07a0abcd3899d6946ac7a58db05bb.png">

This the strategy of how to update the policy. It is derived from Markov decision process and Bellman Equation. The detail of the prove will not be showed at this. The proving procedure is a litter complex but the conclusion and implement of Q-leaning is not very difficult. 

Additional, other algorithm like Sarsa, TD algorithm, are very like with Q-learning except the policy definition and update rules has some difference. 

<div style="page-break-before: always;"> </div>

### Implement

The implement of reinforcement learning is step by step which is start from easy environment to much complicated agent. All agents are from gym platform. I tried 3 of them, which **Taxi-v2, Cartpole-v0, Breakout-ram-v0**. 

**Taxi-v2**

 In this environment the yellow square represents the taxi, the (“|”) represents a wall, the blue letter represents the pick-up location, and the purple letter is the drop-off location. The taxi will turn green when it has a passenger aboard. While we see colours and shapes that represent the environment, the algorithm does not think like us and only understands a flattened state, in this case an integer.
 <div align="center" style="zoom:100%">
<img src ="https://github.com/HawkTom/ML_self_test/blob/master/Reinforcement%20Learning/taxi-v2.PNG?raw=true">
</div>

The details of this environment:
- action_space: 6 possible action (down (0), up (1), right (2), left (3), pick-up (4), and drop-off (5))
- observation_space: 500. Each integer represent a state of environment. 
- one step will return four variables (state number, reward, done, information)

Because the observation is finite and discrete, so the problem become easy to solve. We could get a Q table for the environment. 

Step 1:  Create a Q table which is a policy for the environment. 
```(python)
import numpy as np
Q = np.zeros([env.observation_space.n, env.action_space.n])
```
The size of Q-table based on observation space and action space, so in this instance , the Q table is 500*6

Step 2: implement a very basic Q learning
```python
G = 0 # accumulate reward
alpha = 0.618 # learning rate
for episode in range(1, 1001):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
        action = np.argmax(Q[state]) # 1
        state2, reward, done, info = env.step(action) # 2
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action]) # 3
        G += reward
        state = state2
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))
```
**# 1**: The agent choose an action with the highest Q value fro Q table correspond to its current state

**# 2**: The agent step one action, and the environment will return some information.

**# 3**: According to the reward and the old Q table, we can update the Q table by the formula introduced in last section. 

After so many episodes, the algorithm will converge and determine the optimal action for every state using the Q table, ensuring the highest possible reward. 
the behaviour of algorithm shows below
 <div align="center" style="zoom:70%">
<img src ="https://github.com/HawkTom/ML_self_test/blob/master/Reinforcement%20Learning/Figure_1.png?raw=true">
</div>

Step 3: using Q table replay the game
```python
done = False
state = env.reset()
G = 0;
for i in range(100):
    action = np.argmax(Q[state])
    state, reward, done, info = env.step(action)
    G += reward
    env.render()
    if done:
        print('reward: {}, loops: {}'.format(G, i))
        break
```

Result: from a initial state, it only step 8 actions from starter to end point. The accumulate reward is 12.

**CartPole-v0**

The previous environment is very easy to solve because the observation space is finite and discrete. Now we consider another environment named "CartPole-v0", which is also very simple but a litter more complicated than the "taxi-v2". 

Brief introduction of environment:
A pole is attached by an unactuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
 <div align="center" style="zoom:70%">
<img src = https://i.ytimg.com/vi/S7UDdHtCHAE/hqdefault.jpg>
</div>

The details of this environment:
- action_space: 2 possible action (push cart to left(0) or right(1))
- observation_space: infinite and continues.
- observation representation: (cart position, cart velocity, pole angle, pole velocity at tip) 
- reward is 1 for every step
- Episode length is greater than 200 which means the maximum reward is 200.

Basic idea:

Different from the "taxi-v2" environment, the observation here is continues and infinite, so it is impossible to get a Q-table. After research from the literature, there are two alternative way to solve this situation:
- Deep Q Networks: use neural networks to replace the table. This is a very popular method for many reinforcement learning
- Another way if transform the continues space to discrete space so that we create an Q table to solve the problem

In this lab, I choose the second way to solve it. 

Actually the main challenge was to convert the continuous, 4-dimensional input space to a discrete space with a finite and preferably small, yet expressive, number of discrete states. The less states we have, the smaller the Q-table will be, the less steps the agent will need to properly learn its values. 

Based on my survey in the Internet, refer to other's code, I implement the solution of CartPole-v0.

Step 1: Transforming the feature space
I initially started by scaling theta down to a discrete interval theta ∈ [0, 6] ⊂ ℕ (which is, to be precise, just a set of integers {0..6}) and theta' to theta' ∈ [0, 12] ⊂ ℕ. I dropped the x and x' features completely. 
So, the Q table is (1, 1, 6, 12, 2). The size of Q table represent (position, velocity, angle, angle velocity, action space). As a result, the observation is discrete.

```python
Q = np.zeros((1, 1, 6, 12) + (env.action_space.n,))
def discretize(env, obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)
```

Step 2: implement the Q - learning algorithm
```python
scores = deque(maxlen=100)
for e in range(n_episodes):
    current_state = discretize(env.reset())
    print(current_state)
    alpha = get_alpha(e)
    epsilon = get_epsilon(e)
    done = False

    while not done:
        # env.render()
        action = choose_action(current_state, epsilon)
        obs, reward, done, _ = env.step(action)
        new_state = discretize(obs)
        Q[state_old][action] += alpha * (reward + gamma * np.max(Q[state_new]) - Q[state_old][action])
        current_state = new_state

    scores.append(i)
    mean_score = np.mean(scores)
    if mean_score >= n_win_ticks and e >= 100:
        if not quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
        return e - 100
    if e % 100 == 0 and not quiet:
        print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
```

The code for Q-learning algorithm is same as the last environment whose the key line are both the update of Q-table. So I will not introduce the detail of the algorithm.
 <div align="center" style="zoom:60%">
<img src ="https://github.com/HawkTom/ML_self_test/blob/master/Reinforcement%20Learning/Figure_1-1.png?raw=true">
</div>
We could find from the above reward curves that the environment is very easy to solve by Q-learning table after discrete the observation space.

Step 3: replay the game by trained Q-table
```python
def replay():
    state = discretize(env.reset())
    G = 0;
    for i in range(1000):
        action = choose_action(state, -0.1)
        obs, reward, done, _ = env.step(action)
        state = discretize(obs)
        env.render()
        G += reward
        if done:
            print('Reward is: {}, Tick: {}'.format(G, i))
            break
```
The result:   Reward is 200, Ticks: 199. 
The result means the policy control the cart-pole successfully because after 200 steps, the environment automatically resets itself. 

**Breakout-ram-v0**

This environment is more difficult than the previous two environments. It is an Atari game. The condition become very complicated.
 <div align="center" style="zoom:100%">
<img src ="https://github.com/HawkTom/ML_self_test/blob/master/Reinforcement%20Learning/breakout.PNG?raw=true">
</div>

The details of this environment:
- action_space: 4 possible action 
- observation_space: infinite and continues.
- observation representation: array: 128 
- reward: based on rule. 

So, the challenge of this problem is representation. If we can represent the observation, action and reward, the problem could be easy. Continue from the previous problem, we try to discrete the environment, but it is very difficult because the the dimension of state representation is very large. 

So, the only choice is DQN( deep Q network). However, it needs to design the neural network and adjust the hyper-parameters. At same time, the training time will be very long, it is very hard for us to get a optimal policy for the the environment. 

Here is an image of introduction of how Deep Q network works. 

 <div align="center" style="zoom:100%">
<img src =https://www.researchgate.net/profile/Kao-Shing_Hwang/publication/224352065/figure/fig5/AS:302635006611475@1449165216980/FIGURE-5-Structure-of-the-proposed-stochastic-real-valued-Q-learning.png>
</div>

And I find pseudo code of Q network in the Internet. And because of the time and device limiting, I didn't implement this tough task.  

<div align="center" style="zoom:100%">
<img src ="http://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs40638-016-0055-x/MediaObjects/40638_2016_55_Figa_HTML.gif">
</div>


### Conclusion
Reinforcement learning is very popular learning theory. By implementing an basic algorithm Q-learning, we could understand the reinforcement learning deeply and get the ability of how to use the Q-learning to solve reinforcement learning problem. 