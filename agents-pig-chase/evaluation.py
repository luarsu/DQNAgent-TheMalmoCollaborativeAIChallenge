# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

import os
import sys
import numpy as np
from time import sleep

from common import parse_clients_args, ENV_AGENT_NAMES, ENV_TARGET_NAMES
from agent import PigChaseChallengeAgent, FocusedAgent
from common import ENV_AGENT_NAMES
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))


class PigChaseEvaluator(object):
    def __init__(self, clients, agent_100k, agent_500k, state_builder):
        assert len(clients) >= 2, 'Not enough clients provided'
        batch_size = 32
        self._clients = clients
        self._agent_100k = agent_100k
        self._agent_500k = agent_500k
        self._state_builder = state_builder
        self._accumulators = {'100k': []} ## Structure to store the scores obtained

    def save(self, experiment_name, filepath):
        """
        Save the evaluation results in a JSON file 
        understandable by the leaderboard.
        
        Note: The leaderboard will not accept a submission if you already 
        uploaded a file with the same experiment name.
        
        :param experiment_name: An identifier for the experiment
        :param filepath: Path where to store the results file
        :return: 
        """

        assert experiment_name is not None, 'experiment_name cannot be None'

        from json import dump
        from os.path import exists, join, pardir, abspath
        from os import makedirs
        from numpy import mean, var

        # Compute metrics that store the score
        metrics = {key: {'mean': mean(buffer),
                         'var': var(buffer),
                         'count': len(buffer)}
                   for key, buffer in self._accumulators.items()}

        metrics['experimentname'] = experiment_name
        print(metrics)

    def run(self):
        from multiprocessing import Process

        env = PigChaseEnvironment(self._clients, self._state_builder,
                                  role=1, randomize_positions=True)
        print('==================================')
        print('Starting evaluation of Agent provided')
        ##Initialize the threads with the two agents
        p = Process(target=run_challenge_agent, args=(self._clients,True))
        p.start()
        sleep(5)
        agent_loop(self._agent_100k, env, self._accumulators['100k'], True)
        self._agent_100k.save('pesosBuenos.h5') ##This saves the weights of the neural network when the agent has finished its training
        p.terminate()


def run_challenge_agent(clients, is100):
    builder = PigChaseSymbolicStateBuilder()
    env = PigChaseEnvironment(clients, builder, role=0,
                              randomize_positions=True)
    agent = FocusedAgent(ENV_AGENT_NAMES[0], ENV_TARGET_NAMES[0])
    agent_loop(agent, env, None, is100)

##Method to adapt the information provided by malmo of the perceptions of the agent
def adapt_state(state):
    entities = state[1]
    me_details = [e for e in entities if e['name'] == 'Agent_1'][0]
    xA1 = me_details['x']
    zA1 = me_details['z']
    me_details = [e for e in entities if e['name'] == 'Agent_2'][0]
    xA2 = me_details['x']
    zA2 = me_details['z']
    me_details = [e for e in entities if e['name'] == 'Pig'][0]
    xP = me_details['x']
    zP = me_details['z']
    arr=np.array([xA1,zA1,xA2,zA2,xP,zP])
    arr2=np.array([arr])
    return arr2

##Main loop of the experiment that implements the DQN algorithm
def agent_loop(agent, env, metrics_acc, is100):
    EVAL_EPISODES = 100 ##Defines the number of games the agent is going to play, change if want to make it play more
    accumulators = {'100k': []}
    agent_done = False
    reward = 0 
    count5 = 0 ##Count of the number of times it reaches an individual solution for later analysis
    count25 = 0 ##Count of the number of times it reaches an collective solution for later analysis
    batch_size = 32
    episode = 0
    num_memory=0
    state = env.reset() ##Resets the environment for the game
    from numpy import mean, var
    while episode < EVAL_EPISODES:
        # check if env needs reset
        if env.done:
            ##If the agent has reached an individual/collective solution for the game, add one to the corresponding counter
            if(reward == 24):
                count25= count25+1
            if(reward == 4):
                count5= count5+1
            ##Prints the data obtained after every game
            print('Number of collective solutions obtained: ')
            print(count25)
            print('Number of individual solutions obtained : ')
            print(count5)
            print('Metrics of the score per step:')
            metrics = {key: {'mean': mean(buffer),
                             'var': var(buffer),
                             'count': len(buffer)}
                       for key, buffer in accumulators.items()}
            print(metrics)

            print('Episode %d (%.2f)%%' % (episode, (episode / EVAL_EPISODES) * 100.))
            if num_memory>batch_size:
                print('Entering replay to train agent :'+ agent.name)
                agent.replay(batch_size)
            ##After 10 games it stores the weights of the neural network 
            if(episode%10==0):
                if(agent.name=='Agent_1'):
                    agent.save('pesosBuenos.h5')
                if(agent.name=='Agent_2'):
                    agent.save('pesosBuenos2.h5')
            state = env.reset() ##Reset environment for new game
            while state is None:
                # this can happen if the episode ended with the first
                # action of the other agent
                print('Warning: received state == None.')
                state = env.reset()

            episode += 1

        # select an action
        action = agent.act(state, reward, agent_done, is_training=True)
        # take a step
        next_state, reward, agent_done = env.do(action)
        next_state2=adapt_state(next_state) ##Adaot the state obtained for the neural network
        ##Store  the previous experience in memory for later re-training of the agent
        agent.remember(state, action, reward, next_state2, agent_done)
        state = next_state
        num_memory=num_memory+1
        if(is100):
            accumulators['100k'].append(reward) ##Apend the score obtained per step for later analysis
        if metrics_acc is not None:
            metrics_acc.append(reward)

def printmetrics(experiment_name, metrics_acc):
    assert experiment_name is not None, 'experiment_name cannot be None'

    from json import dump
    from os.path import exists, join, pardir, abspath
    from os import makedirs
    from numpy import mean, var

    # Compute metrics
    metrics = {key: {'mean': mean(buffer),
                     'var': var(buffer),
                     'count': len(buffer)}
               for key, buffer in metrics_acc.items()}

    metrics['experimentname'] = experiment_name
    print(metrics)