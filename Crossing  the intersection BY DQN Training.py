import airsim
import csv
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from random import randint
import random
from cntk.core import Value
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Dense, default_options
from cntk.layers.typing import Signature, Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.train import Trainer

import csv
class ReplayMemory(object):
    """
    ReplayMemory跟踪動態環境。
     我們存儲所有轉換(s(t), action, s(t+1), reward, done).
    重放存儲器使我們能夠從中有效地採樣微型批次，並生成正確的狀態表示
     （不包括所需的前幾幀）.
    """
    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ 返回內存中當前存在的項目數
         返回值：Int> = 0
        """
        return self._count

    def append(self, state, action, reward, done):
        #append附加
        """將指定的過渡追加到內存。
         屬性：
            state (Tensor[sample_shape]): 要附加的狀態
            action (int): 表示已完成操作的整數
            reward (float):一個整數，表示執行此操作可獲得的獎勵
            done (bool):一個布爾值，指定此狀態是否為終端（情節完成）
        """
        #assert宣告
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ 在內存中生成大小隨機整數映射索引。
             可以使用#get_state（）檢索返回的索引。
             如果要直接檢索樣本，請參見方法＃mini-batch（）。
        屬性:
            size (int): The mini-batch size(最小批量大小)
        返回值:
             索引 of the sampled states ([int])
        """

        #局部變量訪問在循環中更快
        #terminals終端
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []
        #len() 方法返回對象項目個數
        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ 使用size參數指定的樣本數量生成一個小批量。
         屬性：
            size (int): Minibatch size
       返回值:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        使用重放存儲器返回指定狀態。 狀態由
         最後的“ history_length”感知。
         屬性：
            index (int): State's index
        Returns:
            State at  指定索引處(Tensor[history_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

class History(object):
    """
    累加器跟踪代理要使用的N個先前幀進行評估
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ 沿第一軸堆疊有N個先前狀態的底層緩衝區
        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history
        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0
        """
        self._buffer.fill(0)

class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy
    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.
        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step
        Attributes:
            step (int)
        """
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore
        Attributes:
            step (int) : Current step
        Returns:
             bool : True if exploring, False otherwise
        """
        return np.random.rand() < self._epsilon(step)

def huber_loss(y, y_hat, delta):
    """ Compute the Huber Loss as part of the model graph
    Huber Loss is more robust to outliers. It is defined as:
     if |y - y_hat| < delta :
        0.5 * (y - y_hat)**2
    else :
        delta * |y - y_hat| - 0.5 * delta**2
    Attributes:
        y (Tensor[-1, 1]): Target value
        y_hat(Tensor[-1, 1]): Estimated value
        delta (float): Outliers threshold
    Returns:
        CNTK Graph Node
    """
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)

    less_than = 0.5 * square(error)
    more_than = (delta * abs_error) - half_delta_squared
    loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)

    return reduce_sum(loss_per_sample, name='loss')

class DeepQAgent(object):
    """
    Implementation of Deep Q Neural Network agent like in:
        Nature 518. "Human-level control through deep reinforcement learning" (Mnih & al. 2015)
    """
    def __init__(self, input_shape, nb_actions,
                 gamma=0.99, explorer=LinearEpsilonAnnealingExplorer(1, 0.1, 1000000),
                 learning_rate=0.00025, momentum=0.95, minibatch_size=32,
                 memory_size=500000, train_after=10000, train_interval=4, target_update_interval=10000,
                 monitor=True):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        self._explorer = explorer
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = ReplayMemory(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

        # Action Value model (used by agent to interact with the environment)
        with default_options(activation=relu, init=he_uniform()):
            self._action_value_net = Sequential([
                Convolution2D((8, 8), 16, strides=4),
                Convolution2D((4, 4), 32, strides=2),
                Convolution2D((3, 3), 32, strides=1),
                Dense(256, init=he_uniform(scale=0.01)),
                Dense(nb_actions, activation=None, init=he_uniform(scale=0.01))
            ])
        self._action_value_net.update_signature(Tensor[input_shape])

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self._target_net = self._action_value_net.clone(CloneMethod.freeze)

        # Function computing Q-values targets as part of the computation graph
        @Function
        @Signature(post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def compute_q_targets(post_states, rewards, terminals):
            return element_select(
                terminals,
                rewards,
                gamma * reduce_max(self._target_net(post_states), axis=0) + rewards,
            )

        # Define the loss, using Huber Loss (more robust to outliers)
        @Function
        @Signature(pre_states=Tensor[input_shape], actions=Tensor[nb_actions],
                   post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def criterion(pre_states, actions, post_states, rewards, terminals):
            # Compute the q_targets
            q_targets = compute_q_targets(post_states, rewards, terminals)

            # actions is a 1-hot encoding of the action done by the agent
            q_acted = reduce_sum(self._action_value_net(pre_states) * actions, axis=0)

            # Define training criterion as the Huber Loss function
            return huber_loss(q_targets, q_acted, 1.0)

        # Adam based SGD
        lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
        m_schedule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)
        l_sgd = adam(self._action_value_net.parameters, lr_schedule,
                     momentum=m_schedule, variance_momentum=vm_schedule)

        self._metrics_writer = TensorBoardProgressWriter(freq=1, log_dir='metrics', model=criterion) if monitor else None
        self._learner = l_sgd
        self._trainer = Trainer(criterion, (criterion, None), l_sgd, self._metrics_writer)

    def act(self, state):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.
        Attributes:
            state (Tensor[input_shape]): The current environment state
        Returns: Int >= 0 : Next action to do
        """
        # Append the state to the short term memory (ie. History)
        self._history.append(state)

        # If policy requires agent to explore, sample random action
        if self._explorer.is_exploring(self._num_actions_taken):
            action = self._explorer(self.nb_actions)
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            q_values = self._action_value_net.eval(
                # Append batch axis with only one sample to evaluate
                env_with_history.reshape((1,) + env_with_history.shape)
            )

            self._episode_q_means.append(np.mean(q_values))
            self._episode_q_stddev.append(np.std(q_values))

            # Return the value maximizing the expected reward
            action = q_values.argmax()

        # Keep track of interval action counter
        self._num_actions_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state
        Attributes:
            old_state (Tensor[input_shape]): Previous environment state
            action (int): Action done by the agent
            reward (float): Reward for doing this action in the old_state environment
            done (bool): Indicate if the action has terminated the environment
        """
        self._episode_rewards.append(reward)

        # If done, reset short term memory (ie. History)
        if done:
            # Plot the metrics through Tensorboard and reset buffers
            if self._metrics_writer is not None:
                self._plot_metrics()
            self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

    def train(self):
        """ 這允許代理進行自我訓練以更好地了解環境動態。
         代理將計算state（t + 1）的預期獎勵
         並據此在步驟t更新預期報酬。
         目標期望是通過目標網絡計算的，這是一個更穩定的版本
         行動價值網絡，以提高培訓的穩定性。
         目標網絡是按常規間隔更新的行動價值網絡的凍結副本。
        """

        agent_step = self._num_actions_taken

        if agent_step >= self._train_after:
            if (agent_step % self._train_interval) == 0:
                pre_states, actions, post_states, rewards, terminals = self._memory.minibatch(self._minibatch_size)

                self._trainer.train_minibatch(
                    self._trainer.loss_function.argument_map(
                        pre_states=pre_states,
                        actions=Value.one_hot(actions.reshape(-1, 1).tolist(), self.nb_actions),
                        post_states=post_states,
                        rewards=rewards,
                        terminals=terminals
                    )
                )

                # Update the Target Network if needed
                if (agent_step % self._target_update_interval) == 0:
                    self._target_net = self._action_value_net.clone(CloneMethod.freeze)

    def _plot_metrics(self):
        """Plot current buffers accumulated values to visualize agent learning
        """
        if len(self._episode_q_means) > 0:
            mean_q = np.asscalar(np.mean(self._episode_q_means))
            self._metrics_writer.write_value('Mean Q per ep.', mean_q, self._num_actions_taken)

        if len(self._episode_q_stddev) > 0:
            std_q = np.asscalar(np.mean(self._episode_q_stddev))
            self._metrics_writer.write_value('Mean Std Q per ep.', std_q, self._num_actions_taken)

        self._metrics_writer.write_value('Sum rewards per ep.', sum(self._episode_rewards), self._num_actions_taken)

def transform_input(responses):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    from PIL import Image
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert('L')) 

    return im_final


def interpret_action(action): 
        while 1:
            min_v=1
            max_v=5
            warning_dist=-4
            global v2,v3,v4,v5,v6,v7,v8,v9,v10,v211,v212,v213,v214,v215,v216,v17,v18,v19,v20,v21,v21,v22,v23,v24,v25
            v11 = randint(min_v, max_v)
            v12 = randint(min_v, max_v)
            v13 = randint(min_v, max_v)
            v14 = randint(min_v, max_v)
            v15 = randint(min_v, max_v)
            v16 = randint(min_v, max_v)
            v2= randint(min_v, max_v)
            v3= randint(min_v, max_v)
            v4= randint(min_v, max_v)
            v5= randint(min_v, max_v)
            v6= randint(min_v, max_v)
            v7= randint(min_v, max_v)
            v8= randint(min_v, max_v)
            v9= randint(min_v, max_v)
            v10= randint(min_v, max_v)
            v211= randint(min_v, max_v)
            v212= randint(min_v, max_v)
            v213= randint(min_v, max_v)
            v214= randint(min_v, max_v)
            v215= randint(min_v, max_v)
            v216= randint(min_v, max_v)
            v17= randint(min_v, max_v)
            v18= randint(min_v, max_v)
            v19= randint(min_v, max_v)
            v20= randint(min_v, max_v)
            v21 = randint(min_v, max_v)
            v22 = randint(min_v, max_v)
            v23 = randint(min_v, max_v)
            v24 = randint(min_v, max_v)
            v25 = randint(min_v, max_v)

            if y2 - y3 < warning_dist and v2 > v3:           
                v2=v3
            if y3 - y4 < warning_dist and v3 > v4:           
                v3=v4
            if y4 - y5 < warning_dist and v4 > v5:           
                v4=v5
            if y5 - y6 < warning_dist and v5 > v6:           
                v5=v6
            if y6 - y7 < warning_dist and v6 > v7:           
                v6=v7
            if y7 - y8 < warning_dist and v7 > v8:           
                v7=v8
            if y8 - y9 < warning_dist and v8 > v9:           
                v8=v9
            if y9 - y10 < warning_dist and v9 > v10:           
                v9=v10
            if y10 - y11 < warning_dist and v10 > v11:           
                v10=v11
            if y11 - y12 < warning_dist and v11 > v12:           
                v11=v12
            if y12 - y13 < warning_dist and v12 > v13:           
                v12=v13
            if y14 - y15 < warning_dist and v14 < v15:           
                v15=v14
            if y15 - y16 < warning_dist and v15 < v16:           
                v16=v15
            if y16 - y17 < warning_dist and v16 < v17:           
                v17=v16
            if y17 - y18 < warning_dist and v17 < v18:           
                v18=v17
            if y18 - y19 < warning_dist and v18 < v19:           
                v19=v18
            if y19 - y20 < warning_dist and v19 < v20:           
                v20=v19
            if y20 - y21 < warning_dist and v20 < v21:           
                v21=v20
            if y21 - y22 < warning_dist and v21 < v22:           
                v22=v21
            if y22 - y23 < warning_dist and v22 < v23:           
                v23=v22
            if y23 - y24 < warning_dist and v23 < v24:           
                v24=v23
            if y24 - y25 < warning_dist and v24 < v25:           
                v25=v24

            if action == 0:
                quad_offset = v11
            elif action == 1:
                quad_offset = v12
            elif action == 2:
                quad_offset = v13
            elif action == 3:
                quad_offset = v14
            elif action == 4:
                quad_offset = v15
            elif action == 5:
                quad_offset = v16
            
            return quad_offset
            break
def compute_reward(collision_info):
    reward=0    
    if collision_info.object_name !="Road_85":
        reward += -200
    if position0[0]+x1==position1[0]+y2 or position0[0]+x1==position1[1]+y3:
        reward += -x1
    if position0[0]+x1==position1[2]+y4 or position0[0]+x1==position1[3]+y5:
        reward += -x1
    if position0[0]+x1==position1[4]+y6 or position0[0]+x1==position1[5]+y7:
        reward += -x1        
    if position0[0]+x1==position1[6]+y8 or position0[0]+x1==position1[7]+y9:
        reward += -x1
    if position0[0]+x1==position1[8]+y10 or position0[0]+x1==position1[9]+y11:
        reward += -x1
    if position0[0]+x1==position1[10]+y12 or position0[0]+x1==position1[11]+y13:
        reward += -x1
    if position0[0]+x1==position2[0]-y14 or position0[0]+x1==position2[1]-y15:
        reward += -x1
    if position0[0]+x1==position2[2]-y16 or position0[0]+x1==position2[3]-y17:
        reward += -x1
    if position0[0]+x1==position2[4]-y18 or position0[0]+x1==position2[5]-y19:
        reward += -x1
    if position0[0]+x1==position2[6]-y20 or position0[0]+x1==position2[7]-y21:
        reward += -x1
    if position0[0]+x1==position2[8]-y22 or position0[0]+x1==position2[9]-y23:
        reward += -x1
    if position0[0]+x1==position2[10]-y24 or position0[0]+x1==position2[11]-y25:
        reward += -x1
    
    reward += -x1
    reward -= (time.time()-ticks)
    '''reward-=area_count'''
    return reward
    
def isDone(reward):
    done = 0
    if  reward<-100 or -x1-position0[0]>=60 :
        done = 1
    
    return done
    
initX = 0
initY = 0
initZ = -5

# connect to the AirSim simulator 
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True,"Drone1")
client.enableApiControl(True,"Drone2")
client.enableApiControl(True,"Drone3")
client.enableApiControl(True,"Drone4")
client.enableApiControl(True,"Drone5")
client.enableApiControl(True,"Drone6")
client.enableApiControl(True,"Drone7")
client.enableApiControl(True,"Drone8")
client.enableApiControl(True,"Drone9")
client.enableApiControl(True,"Drone10")
client.enableApiControl(True,"Drone11")
client.enableApiControl(True,"Drone12")
client.enableApiControl(True,"Drone13")
client.enableApiControl(True,"Drone14")
client.enableApiControl(True,"Drone15")
client.enableApiControl(True,"Drone16")
client.enableApiControl(True,"Drone17")
client.enableApiControl(True,"Drone18")
client.enableApiControl(True,"Drone19")
client.enableApiControl(True,"Drone20")
client.enableApiControl(True,"Drone21")
client.enableApiControl(True,"Drone22")
client.enableApiControl(True,"Drone23")
client.enableApiControl(True,"Drone24")
client.enableApiControl(True,"Drone25")
client.armDisarm(True,"Drone1")
client.armDisarm(True,"Drone2")
client.armDisarm(True,"Drone3")
client.armDisarm(True,"Drone4")
client.armDisarm(True,"Drone5")
client.armDisarm(True,"Drone6")
client.armDisarm(True,"Drone7")
client.armDisarm(True,"Drone8")
client.armDisarm(True,"Drone9")
client.armDisarm(True,"Drone10")
client.armDisarm(True,"Drone11")
client.armDisarm(True,"Drone12")
client.armDisarm(True,"Drone13")
client.armDisarm(True,"Drone14")
client.armDisarm(True,"Drone15")
client.armDisarm(True,"Drone16")
client.armDisarm(True,"Drone17")
client.armDisarm(True,"Drone18")
client.armDisarm(True,"Drone19")
client.armDisarm(True,"Drone20")
client.armDisarm(True,"Drone21")
client.armDisarm(True,"Drone22")
client.armDisarm(True,"Drone23")
client.armDisarm(True,"Drone24")
client.armDisarm(True,"Drone25")

z=-5
count=0

print("Climbing")
F1=client.moveToZAsync(z,velocity=3,vehicle_name="Drone1")
F2=client.moveToZAsync(z,velocity=3,vehicle_name="Drone2")
F3=client.moveToZAsync(z,velocity=3,vehicle_name="Drone3")
F4=client.moveToZAsync(z,velocity=3,vehicle_name="Drone4")
F5=client.moveToZAsync(z,velocity=3,vehicle_name="Drone5")
F6=client.moveToZAsync(z,velocity=3,vehicle_name="Drone6")
F7=client.moveToZAsync(z,velocity=3,vehicle_name="Drone7")
F8=client.moveToZAsync(z,velocity=3,vehicle_name="Drone8")
F9=client.moveToZAsync(z,velocity=3,vehicle_name="Drone9")
F10=client.moveToZAsync(z,velocity=3,vehicle_name="Drone10")
F11=client.moveToZAsync(z,velocity=3,vehicle_name="Drone11")
F12=client.moveToZAsync(z,velocity=3,vehicle_name="Drone12")
F13=client.moveToZAsync(z,velocity=3,vehicle_name="Drone13")
F14=client.moveToZAsync(z,velocity=3,vehicle_name="Drone14")
F15=client.moveToZAsync(z,velocity=3,vehicle_name="Drone15")
F16=client.moveToZAsync(z,velocity=3,vehicle_name="Drone16")
F17=client.moveToZAsync(z,velocity=3,vehicle_name="Drone17")
F18=client.moveToZAsync(z,velocity=3,vehicle_name="Drone18")
F19=client.moveToPositionAsync(-0.5, 0, z,velocity=3,vehicle_name="Drone19")
F20=client.moveToZAsync(z,velocity=3,vehicle_name="Drone20")
F21=client.moveToZAsync(z,velocity=3,vehicle_name="Drone21")
F22=client.moveToZAsync(z,velocity=3,vehicle_name="Drone22")
F23=client.moveToZAsync(z,velocity=3,vehicle_name="Drone23")
F24=client.moveToZAsync(z,velocity=3,vehicle_name="Drone24")
F25=client.moveToZAsync(z,velocity=3,vehicle_name="Drone25") 
F1.join()
F2.join()
F3.join()
F4.join()
F5.join()
F6.join()
F7.join()
F8.join()
F9.join()
F10.join()
F11.join()
F12.join()
F13.join()
F14.join()
F15.join()
F16.join()
F17.join()
F18.join()
F19.join()
F20.join()
F21.join()
F22.join()
F23.join()
F24.join()
F25.join()

position0=random.sample(range (50,60,5), 1)
position0.sort()
position1=random.sample(range (5,130,5), 12)
position1.sort(reverse = True)
position2=random.sample(range (-130,-5,5), 12)
position2.sort(reverse = True)
F1=client.moveToPositionAsync(position0[0]-60, 0, z,velocity=3,vehicle_name="Drone1")
F2=client.moveToPositionAsync(0,position1[0]-120,z,velocity=3,vehicle_name="Drone2")
F3=client.moveToPositionAsync(0,position1[1]-110,z,velocity=3,vehicle_name="Drone3")
F4=client.moveToPositionAsync(0,position1[2]-100,z,velocity=3,vehicle_name="Drone4")
F5=client.moveToPositionAsync(0,position1[3]-90,z,velocity=3,vehicle_name="Drone5")
F6=client.moveToPositionAsync(0,position1[4]-80,z,velocity=3,vehicle_name="Drone6")
F7=client.moveToPositionAsync(0,position1[5]-70,z,velocity=3,vehicle_name="Drone7")
F8=client.moveToPositionAsync(0,position1[6]-60,z,velocity=3,vehicle_name="Drone8")
F9=client.moveToPositionAsync(0,position1[7]-50,z,velocity=3,vehicle_name="Drone9")
F10=client.moveToPositionAsync(0,position1[8]-40,z,velocity=3,vehicle_name="Drone10")
F11=client.moveToPositionAsync(0,position1[9]-30,z,velocity=3,vehicle_name="Drone11")
F12=client.moveToPositionAsync(0,position1[10]-20,z,velocity=3,vehicle_name="Drone12")
F13=client.moveToPositionAsync(0,position1[11]-10,z,velocity=3,vehicle_name="Drone13")
F14=client.moveToPositionAsync(0,position2[0]+10,z,velocity=3,vehicle_name="Drone14")
F15=client.moveToPositionAsync(0,position2[1]+20,z,velocity=3,vehicle_name="Drone15")
F16=client.moveToPositionAsync(0,position2[2]+30,z,velocity=3,vehicle_name="Drone16")
F17=client.moveToPositionAsync(0,position2[3]+40,z,velocity=3,vehicle_name="Drone17")
F18=client.moveToPositionAsync(0,position2[4]+50,z,velocity=3,vehicle_name="Drone18")
F19=client.moveToPositionAsync(-0.5,position2[5]+60,z,velocity=3,vehicle_name="Drone19")
F20=client.moveToPositionAsync(0,position2[6]+70,z,velocity=3,vehicle_name="Drone20")
F21=client.moveToPositionAsync(0,position2[7]+80,z,velocity=3,vehicle_name="Drone21")
F22=client.moveToPositionAsync(0,position2[8]+90,z,velocity=3,vehicle_name="Drone22")
F23=client.moveToPositionAsync(0,position2[9]+100,z,velocity=3,vehicle_name="Drone23")
F24=client.moveToPositionAsync(0,position2[10]+110,z,velocity=3,vehicle_name="Drone24")
F25=client.moveToPositionAsync(0,position2[11]+120,z,velocity=3,vehicle_name="Drone25")
F1.join()
F2.join()
F3.join()
F4.join()
F5.join()
F6.join()
F7.join()
F8.join()
F9.join()
F10.join()
F11.join()
F12.join()
F13.join()
F14.join()
F15.join()
F16.join()
F17.join()
F18.join()
F19.join()
F20.join()
F21.join()
F22.join()
F23.join()
F24.join()
F25.join()

# Make RL agent
NumBufferFrames = 4
SizeRows = 84
SizeCols = 84
NumActions = 6
agent = DeepQAgent((NumBufferFrames, SizeRows, SizeCols), NumActions, monitor=True)

# Train
epoch = 100
current_step = 0
max_steps = epoch * 250000

responses = client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)])
current_state = transform_input(responses)

collision_count=0
ticks = time.time()
step=0

path = "dqn_data1113.csv"
with open(path, "a+", newline="") as csvfile:
    writer = csv.writer(csvfile)      
    writer.writerow([str("current_step"),str("step"),str("time.time()-ticks"),str("x1"),str("y2"),str("y3"),str("y4"),str("y5"),str("y6"),str("y7"),str("action"),str("reward"),str("done"),
                         str("collision_info.object_name"),str("collision_info.position.x_val"),str("collision_info.position.y_val")])
while True:
    duration=1
    action = agent.act(current_state)
    dp1 = client.simGetVehiclePose(vehicle_name="Drone1")
    dp2 = client.simGetVehiclePose(vehicle_name="Drone2")
    dp3 = client.simGetVehiclePose(vehicle_name="Drone3")
    dp4 = client.simGetVehiclePose(vehicle_name="Drone4")
    dp5 = client.simGetVehiclePose(vehicle_name="Drone5")
    dp6 = client.simGetVehiclePose(vehicle_name="Drone6")
    dp7 = client.simGetVehiclePose(vehicle_name="Drone7")
    dp8 = client.simGetVehiclePose(vehicle_name="Drone8")
    dp9 = client.simGetVehiclePose(vehicle_name="Drone9")
    dp10 = client.simGetVehiclePose(vehicle_name="Drone10")
    dp11 = client.simGetVehiclePose(vehicle_name="Drone11")
    dp12 = client.simGetVehiclePose(vehicle_name="Drone12")
    dp13 = client.simGetVehiclePose(vehicle_name="Drone13")
    dp14 = client.simGetVehiclePose(vehicle_name="Drone14")
    dp15 = client.simGetVehiclePose(vehicle_name="Drone15")
    dp16 = client.simGetVehiclePose(vehicle_name="Drone16")
    dp17 = client.simGetVehiclePose(vehicle_name="Drone17")
    dp18 = client.simGetVehiclePose(vehicle_name="Drone18")
    dp19 = client.simGetVehiclePose(vehicle_name="Drone19")
    dp20 = client.simGetVehiclePose(vehicle_name="Drone20")
    dp21 = client.simGetVehiclePose(vehicle_name="Drone21")
    dp22 = client.simGetVehiclePose(vehicle_name="Drone22")
    dp23 = client.simGetVehiclePose(vehicle_name="Drone23")
    dp24 = client.simGetVehiclePose(vehicle_name="Drone24")
    dp25 = client.simGetVehiclePose(vehicle_name="Drone25")
    global x1,y2,y3,y4,y5,y6,y7,x8,y9,y10,y11,y12,y13,y14,x15,y16,y17,y18,y19,y20,y21,x22,y23,y24,y25
    x1=dp1.position.x_val
    y2=dp2.position.y_val
    y3=dp3.position.y_val
    y4=dp4.position.y_val
    y5=dp5.position.y_val
    y6=dp6.position.y_val
    y7=dp7.position.y_val
    y8=dp8.position.y_val
    y9=dp9.position.y_val
    y10=dp10.position.y_val
    y11=dp11.position.y_val
    y12=dp12.position.y_val
    y13=dp13.position.y_val
    y14=dp14.position.y_val
    y15=dp15.position.y_val
    y16=dp16.position.y_val
    y17=dp17.position.y_val
    y18=dp18.position.y_val
    y19=dp19.position.y_val
    y20=dp20.position.y_val
    y21=dp21.position.y_val
    y22=dp22.position.y_val
    y23=dp23.position.y_val
    y24=dp24.position.y_val
    y25=dp25.position.y_val
    
    quad_offset = interpret_action(action)
    quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    
    global position0,position1,position2
    position0=random.sample(range (50,60,5), 1)
    position0.sort()
    position1=random.sample(range (5,130,5), 12)
    position1.sort(reverse = True)
    position2=random.sample(range (-130,-5,5), 12)
    position2.sort(reverse = True)
    F1 = client.moveByVelocityZAsync(-quad_offset,0,z,duration,vehicle_name="Drone1")
    F2 = client.moveByVelocityZAsync(0,-v2 ,z, duration,vehicle_name="Drone2")
    F3 = client.moveByVelocityZAsync(0,-v3 ,z, duration,vehicle_name="Drone3")
    F4 = client.moveByVelocityZAsync(0,-v4 ,z, duration,vehicle_name="Drone4")
    F5 = client.moveByVelocityZAsync(0,-v5 ,z, duration,vehicle_name="Drone5")
    F6 = client.moveByVelocityZAsync(0,-v6 ,z, duration,vehicle_name="Drone6")
    F7 = client.moveByVelocityZAsync(0,-v7 ,z, duration,vehicle_name="Drone7")
    F8 = client.moveByVelocityZAsync(0,-v8 ,z, duration,vehicle_name="Drone8")
    F9 = client.moveByVelocityZAsync(0,-v9 ,z, duration,vehicle_name="Drone9")
    F10 = client.moveByVelocityZAsync(0,-v10 ,z, duration,vehicle_name="Drone10")
    F11 = client.moveByVelocityZAsync(0,-v211 ,z, duration,vehicle_name="Drone11")
    F12 = client.moveByVelocityZAsync(0,-v212 ,z, duration,vehicle_name="Drone12")
    F13 = client.moveByVelocityZAsync(0,-v213 ,z, duration,vehicle_name="Drone13")
    F14 = client.moveByVelocityZAsync(0,v214 ,z, duration,vehicle_name="Drone14")
    F15 = client.moveByVelocityZAsync(0,v215 ,z, duration,vehicle_name="Drone15")
    F16 = client.moveByVelocityZAsync(0,v216 ,z, duration,vehicle_name="Drone16")
    F17 = client.moveByVelocityZAsync(0,v17 ,z, duration,vehicle_name="Drone17")
    F18 = client.moveByVelocityZAsync(0,v18 ,z, duration,vehicle_name="Drone18")
    F19 = client.moveByVelocityZAsync(0,v19 ,z, duration,vehicle_name="Drone19")
    F20 = client.moveByVelocityZAsync(0,v20 ,z, duration,vehicle_name="Drone20")
    F21 = client.moveByVelocityZAsync(0,v21 ,z, duration,vehicle_name="Drone21")
    F22 = client.moveByVelocityZAsync(0,v22 ,z, duration,vehicle_name="Drone22")
    F23 = client.moveByVelocityZAsync(0,v23 ,z, duration,vehicle_name="Drone23")
    F24 = client.moveByVelocityZAsync(0,v24 ,z, duration,vehicle_name="Drone24")
    F25 = client.moveByVelocityZAsync(0,v25 ,z, duration,vehicle_name="Drone25")
    F1.join()
    F2.join()
    F3.join()
    F4.join()
    F5.join()
    F6.join()
    F7.join()
    F8.join()
    F9.join()
    F10.join()
    F11.join()
    F12.join()
    F13.join()
    F14.join()
    F15.join()
    F16.join()
    F17.join()
    F18.join()
    F19.join()
    F20.join()
    F21.join()
    F22.join()
    F23.join()
    F24.join()
    F25.join()    
    
    global area_count
    area_count=0
    if position0[0]+x1<=-5:
        if -5<=(position1[0]+y2)<=5:
            area_count+=1
            pass
        if -5<=(position1[1]+y3)<=5:
            area_count+=1
            pass
        if -5<=(position1[2]+y4)<=5:
            area_count+=1
            pass
        if -5<=(position1[3]+y5)<=5:
            area_count+=1
            pass
        if -5<=(position1[4]+y6)<=5:
            area_count+=1
            pass
        if -5<=(position1[5]+y7)<=5:
            area_count+=1
            pass
        if -5<=(position1[6]+y8)<=5:
            area_count+=1
            pass
        if -5<=(position1[7]+y9)<=5:
            area_count+=1
            pass
        if -5<=(position1[8]+y10)<=5:
            area_count+=1
            pass
        if -5<=(position1[9]+y11)<=5:
            area_count+=1
            pass
        if -5<=(position1[10]+y12)<=5:
            area_count+=1
            pass
        if -5<=(position1[11]+y13)<=5:
            area_count+=1
            pass
        if -5<=-(position2[0]+y14)<=5:
            area_count+=1
            pass
        if -5<=-(position2[1]+y15)<=5:
            area_count+=1
            pass
        if -5<=-(position2[2]+y16)<=5:
            area_count+=1
            pass
        if -5<=-(position2[3]+y17)<=5:
            area_count+=1
            pass
        if -5<=-(position2[4]+y18)<=5:
            area_count+=1
            pass
        if -5<=-(position2[5]+y19)<=5:
            area_count+=1
            pass
        if -5<=-(position2[6]+y20)<=5:
            area_count+=1
            pass
        if -5<=-(position2[7]+y21)<=5:
            area_count+=1
            pass
        if -5<=-(position2[8]+y22)<=5:
            area_count+=1
            pass
        if -5<=-(position2[9]+y23)<=5:
            area_count+=1
            pass
        if -5<=-(position2[10]+y24)<=5:
            area_count+=1
            pass
        if -5<=-(position2[11]+y25)<=5:
            area_count+=1
            pass
        '''
        print("area_count=",area_count)
        print("position0",position0[0])
        print(x1)
        print("position1",position1)
        print("position2",position2)
        print(x1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25)
        '''
    quad_state = client.getMultirotorState().kinematics_estimated.position
    quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    collision_info = client.simGetCollisionInfo()
    reward = compute_reward(collision_info)
    done = isDone(reward)

    agent.observe(current_state, action, reward, done)
    agent.train()
    #csv輸出
    path = "dqn_data1113.csv"
    with open(path, "a+", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([str(current_step),str(step),str(time.time()-ticks),str(x1),str(y2),str(y3),str(y4),str(y5),str(y6),str(y7),str(quad_offset),str(reward),str(done),
                         str(collision_info.object_name),str(collision_info.position.x_val),str(collision_info.position.y_val)])
        
    step+=1

    #重置設定!!!!
    if done:
        
        client.armDisarm(False)
        client.reset()
        time.sleep(0.5)
        
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True,"Drone1")
        client.enableApiControl(True,"Drone2")
        client.enableApiControl(True,"Drone3")
        client.enableApiControl(True,"Drone4")
        client.enableApiControl(True,"Drone5")
        client.enableApiControl(True,"Drone6")
        client.enableApiControl(True,"Drone7")
        client.enableApiControl(True,"Drone8")
        client.enableApiControl(True,"Drone9")
        client.enableApiControl(True,"Drone10")
        client.enableApiControl(True,"Drone11")
        client.enableApiControl(True,"Drone12")
        client.enableApiControl(True,"Drone13")
        client.enableApiControl(True,"Drone14")
        client.enableApiControl(True,"Drone15")
        client.enableApiControl(True,"Drone16")
        client.enableApiControl(True,"Drone17")
        client.enableApiControl(True,"Drone18")
        client.enableApiControl(True,"Drone19")
        client.enableApiControl(True,"Drone20")
        client.enableApiControl(True,"Drone21")
        client.enableApiControl(True,"Drone22")
        client.enableApiControl(True,"Drone23")
        client.enableApiControl(True,"Drone24")
        client.enableApiControl(True,"Drone25")
        
        client.armDisarm(True,"Drone1")
        client.armDisarm(True,"Drone2")
        client.armDisarm(True,"Drone3")
        client.armDisarm(True,"Drone4")
        client.armDisarm(True,"Drone5")
        client.armDisarm(True,"Drone6")
        client.armDisarm(True,"Drone7")
        client.armDisarm(True,"Drone8")
        client.armDisarm(True,"Drone9")
        client.armDisarm(True,"Drone10")
        client.armDisarm(True,"Drone11")
        client.armDisarm(True,"Drone12")
        client.armDisarm(True,"Drone13")
        client.armDisarm(True,"Drone14")
        client.armDisarm(True,"Drone15")
        client.armDisarm(True,"Drone16")
        client.armDisarm(True,"Drone17")
        client.armDisarm(True,"Drone18")
        client.armDisarm(True,"Drone19")
        client.armDisarm(True,"Drone20")
        client.armDisarm(True,"Drone21")
        client.armDisarm(True,"Drone22")
        client.armDisarm(True,"Drone23")
        client.armDisarm(True,"Drone24")
        client.armDisarm(True,"Drone25")       
        
        F1=client.moveToPositionAsync(position0[0]-60, 0, z,velocity=3,vehicle_name="Drone1")
        F2=client.moveToPositionAsync(0,position1[0]-120,z,velocity=3,vehicle_name="Drone2")
        F3=client.moveToPositionAsync(0,position1[1]-110,z,velocity=3,vehicle_name="Drone3")
        F4=client.moveToPositionAsync(0,position1[2]-100,z,velocity=3,vehicle_name="Drone4")
        F5=client.moveToPositionAsync(0,position1[3]-90,z,velocity=3,vehicle_name="Drone5")
        F6=client.moveToPositionAsync(0,position1[4]-80,z,velocity=3,vehicle_name="Drone6")
        F7=client.moveToPositionAsync(0,position1[5]-70,z,velocity=3,vehicle_name="Drone7")
        F8=client.moveToPositionAsync(0,position1[6]-60,z,velocity=3,vehicle_name="Drone8")
        F9=client.moveToPositionAsync(0,position1[7]-50,z,velocity=3,vehicle_name="Drone9")
        F10=client.moveToPositionAsync(0,position1[8]-40,z,velocity=3,vehicle_name="Drone10")
        F11=client.moveToPositionAsync(0,position1[9]-30,z,velocity=3,vehicle_name="Drone11")
        F12=client.moveToPositionAsync(0,position1[10]-20,z,velocity=3,vehicle_name="Drone12")
        F13=client.moveToPositionAsync(0,position1[11]-10,z,velocity=3,vehicle_name="Drone13")
        F14=client.moveToPositionAsync(0,position2[0]+10,z,velocity=3,vehicle_name="Drone14")
        F15=client.moveToPositionAsync(0,position2[1]+20,z,velocity=3,vehicle_name="Drone15")
        F16=client.moveToPositionAsync(0,position2[2]+30,z,velocity=3,vehicle_name="Drone16")
        F17=client.moveToPositionAsync(0,position2[3]+40,z,velocity=3,vehicle_name="Drone17")
        F18=client.moveToPositionAsync(0,position2[4]+50,z,velocity=3,vehicle_name="Drone18")
        F19=client.moveToPositionAsync(-0.5,position2[5]+60,z,velocity=3,vehicle_name="Drone19")
        F20=client.moveToPositionAsync(0,position2[6]+70,z,velocity=3,vehicle_name="Drone20")
        F21=client.moveToPositionAsync(0,position2[7]+80,z,velocity=3,vehicle_name="Drone21")
        F22=client.moveToPositionAsync(0,position2[8]+90,z,velocity=3,vehicle_name="Drone22")
        F23=client.moveToPositionAsync(0,position2[9]+100,z,velocity=3,vehicle_name="Drone23")
        F24=client.moveToPositionAsync(0,position2[10]+110,z,velocity=3,vehicle_name="Drone24")
        F25=client.moveToPositionAsync(0,position2[11]+120,z,velocity=3,vehicle_name="Drone25")
        F1.join()
        F2.join()
        F3.join()
        F4.join()
        F5.join()
        F6.join()
        F7.join()
        F8.join()
        F9.join()
        F10.join()
        F11.join()
        F12.join()
        F13.join()
        F14.join()
        F15.join()
        F16.join()
        F17.join()
        F18.join()
        F19.join()
        F20.join()
        F21.join()
        F22.join()
        F23.join()
        F24.join()
        F25.join()
        
        plt.plot(current_step,reward)
        plt.xlabel('step')
        plt.ylabel('reward')
        fig = plt.figure()
        plt.show
        fig.savefig(r'C:\Users\AILab\Desktop\0920電腦備份\wang2\dqn撒旦\plt\plot.png')
        
        current_step +=1
        print('current_step',current_step)
        if current_step==100000:
            print('the end')
            break
        step=0
        ticks = time.time()
    responses = client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)],vehicle_name="Drone1")
    current_state = transform_input(responses)