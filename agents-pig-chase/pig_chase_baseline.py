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

import numpy as np
import os
import sys

from argparse import ArgumentParser
from datetime import datetime

import six
from os import path
from threading import Thread, active_count
from time import sleep

from malmopy.agent import RandomAgent
try:
    from malmopy.visualization.tensorboard import TensorboardVisualizer
    from malmopy.visualization.tensorboard.cntk import CntkConverter
except ImportError:
    print('Cannot import tensorboard, using ConsoleVisualizer.')
    from malmopy.visualization import ConsoleVisualizer

from common import parse_clients_args, visualize_training, ENV_AGENT_NAMES, ENV_TARGET_NAMES
from agent import PigChaseChallengeAgent, FocusedAgent
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder

# Enforce path
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

BASELINES_FOLDER = 'results/baselines/pig_chase/%s/%s'
EPOCH_SIZE = 100


def agent_factory(name, role, baseline_agent, clients, max_epochs,
                  logdir, visualizer):

    assert len(clients) >= 2, 'Not enough clients (need at least 2)'
    clients = parse_clients_args(clients)
    batch_size = 32

    builder = PigChaseSymbolicStateBuilder()
    env = PigChaseEnvironment(clients, builder, role=role,
                              randomize_positions=True)

    if role == 0:
        agent = PigChaseChallengeAgent(name)

        if type(agent.current_agent) == RandomAgent:
            agent_type = PigChaseEnvironment.AGENT_TYPE_1
        else:
            agent_type = PigChaseEnvironment.AGENT_TYPE_2
        ##Aqui el state hay que modificarlo para que se adapte a lo que la red neurnal necesita
        state = env.reset(agent_type)

        reward = 0
        agent_done = False
        num_actions=0
        while True:


            # take a step
            

            # reset if needed
            if env.done:
                print(agent.check_memory(batch_size))
                if type(agent.current_agent) == RandomAgent:
                    agent_type = PigChaseEnvironment.AGENT_TYPE_1
                else:
                    agent_type = PigChaseEnvironment.AGENT_TYPE_2
                ##Aqui el state habria que modificarlo de nuevo

                if num_actions > batch_size:
                    print('Entrando a replay 1')
                    agent.replay(batch_size)
                state = env.reset(agent_type)
            
            # select an action
            #print('Accion del role 1')
            action = agent.act(state, reward, agent_done, is_training=True)
            next_state, reward, agent_done = env.do(action)
            num_actions=num_actions+1
            next_state2=adapt_state(next_state)
            agent.remember(state, action, reward, next_state2, agent_done)
            ##Aqui state= obs (que seria el estado anterior estado modificado)
            state = next_state
        ##No estoy seguro de si esto va aqui por el while true (no se cuando acaba). Deberia ir cuando acaba una partida
        ##Hacer check si hace el replay o no. Si no lo hace nunca, meter el replay dentro de el if(env.done (signifca que una etapa ha acabado y empieza otra, por lo que deberia esta bien))

           
            


    else:

        if baseline_agent == 'astar':
            agent = FocusedAgent(name, ENV_TARGET_NAMES[0])
        else:
            agent = RandomAgent(name, env.available_actions)

        state = env.reset()
        reward = 0
        agent_done = False
        viz_rewards = []

        max_training_steps = EPOCH_SIZE * max_epochs
        for step in six.moves.range(1, max_training_steps+1):

            # check if env needs reset
            if env.done:

                visualize_training(visualizer, step, viz_rewards)
                viz_rewards = []
                ##No se si esto se tiene que hacer tambien aqui o no, hacer check
                if agent.check_memory(batch_size)>batch_size:
                    print('Entrando a replay 2')
                    agent.replay(batch_size)
                state = env.reset()

            # select an action
            #print('Accion del role 2')
            action = agent.act(state, reward, agent_done, is_training=True)
            # take a step
            next_state, reward, agent_done = env.do(action)
            next_state2=adapt_state(next_state)
            agent.remember(state, action, reward, next_state2, agent_done)
            ##Aqui state= obs (que seria el estado anterior estado modificado)
            state = next_state
            #obs, reward, agent_done = env.do(action)
            viz_rewards.append(reward)

            agent.inject_summaries(step)



def run_experiment(agents_def):
    assert len(agents_def) == 2, 'Not enough agents (required: 2, got: %d)'\
                % len(agents_def)

    processes = []
    for agent in agents_def:
        p = Thread(target=agent_factory, kwargs=agent)
        p.daemon = True
        p.start()

        # Give the server time to start
        if agent['role'] == 0:
            sleep(1)

        processes.append(p)

    try:
        # wait until only the challenge agent is left
        while active_count() > 2:
            sleep(0.1)
    except KeyboardInterrupt:
        print('Caught control-c - shutting down.')
def adapt_state(state):
    entities = state[1]
    ##Ejemplo de como se sacan los datos del propio agente.
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

if __name__ == '__main__':
    batch_size = 32
    arg_parser = ArgumentParser('Pig Chase baseline experiment')
    arg_parser.add_argument('-t', '--type', type=str, default='astar',
                            choices=['astar', 'random'],
                            help='The type of baseline to run.')
    arg_parser.add_argument('-e', '--epochs', type=int, default=5,
                            help='Number of epochs to run.')
    arg_parser.add_argument('clients', nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Minecraft clients endpoints (ip(:port)?)+')
    args = arg_parser.parse_args()

    logdir = BASELINES_FOLDER % (args.type, datetime.utcnow().isoformat())
    if 'malmopy.visualization.tensorboard' in sys.modules:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)
    else:
        visualizer = ConsoleVisualizer()

    agents = [{'name': agent, 'role': role, 'baseline_agent': args.type,
               'clients': args.clients, 'max_epochs': args.epochs,
               'logdir': logdir, 'visualizer': visualizer}
              for role, agent in enumerate(ENV_AGENT_NAMES)]
    ##print(agents)
     
    run_experiment(agents)

