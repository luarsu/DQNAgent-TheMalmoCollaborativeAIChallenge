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

from __future__ import division

import sys
import time
from collections import namedtuple
from tkinter import ttk, Canvas, W

import numpy as np
import random
import h5py
from keras.layers import Dense
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from common import visualize_training, Entity, ENV_TARGET_NAMES, ENV_ENTITIES, ENV_AGENT_NAMES, \
    ENV_ACTIONS, ENV_CAUGHT_REWARD, ENV_BOARD_SHAPE
from six.moves import range

from malmopy.agent import AStarAgent
from malmopy.agent import QLearnerAgent, BaseAgent, RandomAgent
from malmopy.agent.gui import GuiAgent
from numpy import mean, var

P_FOCUSED = .75
CELL_WIDTH = 33


class PigChaseQLearnerAgent(QLearnerAgent):
    """A thin wrapper around QLearnerAgent that normalizes rewards to [-1,1]"""

    def act(self, state, reward, done, is_training=False):

        reward /= ENV_CAUGHT_REWARD
        return super(PigChaseQLearnerAgent, self).act(state, reward, done,
                                                      is_training)


class PigChaseChallengeAgent(BaseAgent):
    """Pig Chase challenge agent - behaves focused or random."""


    def __init__(self, name, visualizer=None):
        print('Base agent en marcha')
        nb_actions = len(ENV_ACTIONS)
        super(PigChaseChallengeAgent, self).__init__(name, nb_actions,
                                                     visualizer = visualizer)

        self._agents = []
        """Creation of a the agent I programmed giving the pig as a target"""
        self._agents.append(FocusedAgent(name, ENV_TARGET_NAMES[0],
                                         visualizer = visualizer))
        """Adds the reference to the agent created"""
        self.current_agent = self._agents[0]
        
                

    def _select_agent(self, p_focused):
        return self._agents[np.random.choice(range(len(self._agents)),
                                             p = [p_focused, 1. - p_focused])]

    def act(self, new_state, reward, done, is_training=False):
        if done:
            self.current_agent = self._agents[0]
        return self.current_agent.act(new_state, reward, done, is_training)

    def remember(self, state, action, reward, next_state, done):
        self.current_agent.remember(state, action, reward, next_state, done)

    def replay(self, batch_size):
        self.current_agent.replay(batch_size)

    def adapt_state(self, state):
       self.current_agent.adapt_state(state)
    def check_memory(self, batch_size):
        self.current_agent.check_memory(batch_size)

    def save(self, out_dir):
        self.current_agent.save(out_dir)

    def load(self, out_dir):
        self.current_agent(out_dir)

    def inject_summaries(self, idx):
        self.current_agent.inject_summaries(idx)


class FocusedAgent(AStarAgent):
    ACTIONS = ENV_ACTIONS #Allowed actions of the agent
    #Initialization of the agent, creating and defining the parameters of the algorithm, the neural network and the data base to store the agents experiences
    def __init__(self, name, target, visualizer = None):
        print('Agente Luis en marcha')
        super(FocusedAgent, self).__init__(name, len(FocusedAgent.ACTIONS),
                                           visualizer = visualizer)
        self._target = str(target)
        self.state_size = 6 ##Input nodes of the neural network, one per data of state (x and z of the pig and the two agents)
        self.action_size = 3 ##Output nodes of the neural network, one per action possible (turn right, turn left and move forward)
        self.memory = deque(maxlen=5000) #Creation of the data base that stores the experiences of the agent
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.995 #Percentage of exploration decay as the agent learns
        self.learning_rate = 0.001 #Learning rate
        self.qvalue= {'MaxQprediction': []} #Stores the Q value of the prediction of the neural network for later analysis
        self.qvalue2= {'ValorLossQ': []} #Stores the Loss value of the neural network for later analysis
        self.model = self._build_model() #Calls the function that creates the neural network

        ##If you wish to start using an agent that is trained, use the following to load the saved weights into your neural network
        """
        if(name=='Agent_1'):
            self.load('yourneuralnetwork1.h5')
            print('loaded weights for the network 1')
        else:
            self.load('yourneuralnetwork2.h5')
            print('loaded weights for the network 2')
        """

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # Sequential() creates the foundation of the layers.
        model = Sequential()
        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(6) and Hidden Layer with 24 nodes
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # Hidden layer with 24 nodes
        model.add(Dense(24, activation='relu'))
        # Output Layer with # of actions: 3 nodes (move 1, turn 1, turn -1)
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state, reward, done, is_training=False):
        state=self.adapt_state(state) ##Adapts the state received from malmo so it only provides the important information to the neural network
        ##Chooses a random action if less than the exploration rate
        if np.random.rand() <= self.epsilon:
            print('Random action from agent: '+ self.name)
            return random.randrange(self.action_size)
        ##Else, it uses the neural network to predict which action is the best to take
        act_values = self.model.predict(state)
        print('PREDICTED ction from agent:  '+ self.name)
        self.qvalue['MaxQprediction'].append(np.amax(act_values[0])) #Stores the Q value for later analysis
        return np.argmax(act_values[0])  # returns action to take
    
    ##Checks the size of the data base that stores the experiences of the agent
    def check_memory(self, batch_size):
        return(len(self.memory))

    #Method to store the experiences of the agents in memory for later use
    def remember(self, state, action, reward, next_state, done):
        state=self.adapt_state(state)
        ##done is just a boolean that indicates if the state is the final state.
        self.memory.append((state, action, reward, next_state, done))
        
    ##A method that trains the neural net with experiences in the memory is called replay(). 
    ##First, we sample some experiences from the memory and call them minibath.
    def replay(self, batch_size):
        ##Obtener datos de la maxima prediccion Q
        
        print('Max value of Q prediction :')
        prediccionQ = {key: {'mean': mean(buffer),
                         'var': var(buffer),
                         'count': len(buffer)}
                   for key, buffer in self.qvalue.items()}
        print(prediccionQ)
        
        minibatch = random.sample(self.memory, batch_size) ##Takes random samples from memory to re-train the agent
        for state, action, reward, next_state, done in minibatch: ##With it experience from the minibatch the agent is trained
            target = reward
            if not done: ##If the action wasnt the last of the game
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            ##Dos acciones siguientes incluidas para sacar datos del loss
            targetminus=(target-np.amax(target_f[0]))*(target-np.amax(target_f[0]))
            self.qvalue2['ValorLossQ'].append(targetminus) ##Stores the loss value for later analysis
            target_f[0][action] = target ##Uses all previous actions to readjust the weights of the neural network so it makes better predictions
            self.model.fit(state, target_f, epochs=1, verbose=0) ##Calls the keras function to fit the weight values of the network with the data calculated
        
        print('Valor loss:')
        prediccionQ2 = {key: {'mean': mean(buffer),
                         'var': var(buffer),
                         'count': len(buffer)}
                   for key, buffer in self.qvalue2.items()}
        print(prediccionQ2) ##Print the obtained values
        
        if self.epsilon > self.epsilon_min: ##Reduce the exploration rate so every time it performs more predicted actions instead of random ones
            self.epsilon *= self.epsilon_decay

    ##Function to load the weights to the neural network
    def load(self, name):
        self.model.load_weights(name)

    ##Function to save the weights of the trained neural network for later use
    def save(self, name):
        self.model.save_weights(name)
        print('EPSILON:')
        print(self.epsilon)
    ##Method that adapts the data obtained from malmo to give only the important one to the neural network
    def adapt_state(self, state):
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
##The rest of the classes are the ones that came by default with the challenge files
class PigChaseHumanAgent(GuiAgent):
    def __init__(self, name, environment, keymap, max_episodes, max_actions,
                 visualizer, quit):
        self._max_episodes = max_episodes
        self._max_actions = max_actions
        self._action_taken = 0
        self._episode = 1
        self._scores = []
        self._rewards = []
        self._episode_has_ended = False
        self._episode_has_started = False
        self._quit_event = quit
        super(PigChaseHumanAgent, self).__init__(name, environment, keymap,
                                                 visualizer=visualizer)

    def _build_layout(self, root):
        # Left part of the GUI, first person view
        self._first_person_header = ttk.Label(root, text='First Person View', font=(None, 14, 'bold')) \
            .grid(row=0, column=0)
        self._first_person_view = ttk.Label(root)
        self._first_person_view.grid(row=1, column=0, rowspan=10)

        # Right part, top
        self._first_person_header = ttk.Label(root, text='Symbolic View', font=(None, 14, 'bold')) \
            .grid(row=0, column=1)
        self._symbolic_view = Canvas(root)
        self._symbolic_view.configure(width=ENV_BOARD_SHAPE[0]*CELL_WIDTH,
                                      height=ENV_BOARD_SHAPE[1]*CELL_WIDTH)
        self._symbolic_view.grid(row=1, column=1)

        # Bottom information
        self._information_panel = ttk.Label(root, text='Game stats', font=(None, 14, 'bold'))
        self._current_episode_lbl = ttk.Label(root, text='Episode: 0', font=(None, 12))
        self._cum_reward_lbl = ttk.Label(root, text='Score: 0', font=(None, 12, 'bold'))
        self._last_action_lbl = ttk.Label(root, text='Previous action: None', font=(None, 12))
        self._action_done_lbl = ttk.Label(root, text='Actions taken: 0', font=(None, 12))
        self._action_remaining_lbl = ttk.Label(root, text='Actions remaining: 0', font=(None, 12))

        self._information_panel.grid(row=2, column=1)
        self._current_episode_lbl.grid(row=3, column=1, sticky=W, padx=20)
        self._cum_reward_lbl.grid(row=4, column=1, sticky=W, padx=20)
        self._last_action_lbl.grid(row=5, column=1, sticky=W, padx=20)
        self._action_done_lbl.grid(row=6, column=1, sticky=W, padx=20)
        self._action_remaining_lbl.grid(row=7, column=1, sticky=W, padx=20)
        self._overlay = None

        # Main rendering callback
        self._pressed_binding = root.bind('<Key>', self._on_key_pressed)
        self._user_pressed_enter = False

        # UI Update callback
        root.after(self._tick, self._poll_frame)
        root.after(1000, self._on_episode_start)

        root.focus()

    def _draw_arrow(self, yaw, x, y, cell_width, colour):
        if yaw == 0.:
            x1, y1 = (x + .15) * cell_width, (y + .15) * cell_width
            x2, y2 = (x + .5) * cell_width, (y + .4) * cell_width
            x3, y3 = (x + .85) * cell_width, (y + .85) * cell_width

            self._symbolic_view.create_polygon(x1, y1, x2, y3, x3, y1, x2, y2, fill=colour)
        elif yaw == 90.:
            x1, y1 = (x + .15) * cell_width, (y + .15) * cell_width
            x2, y2 = (x + .6) * cell_width, (y + .5) * cell_width
            x3, y3 = (x + .85) * cell_width, (y + .85) * cell_width

            self._symbolic_view.create_polygon(x1, y2, x3, y1, x2, y2, x3, y3, fill=colour)
        elif yaw == 180.:
            x1, y1 = (x + .15) * cell_width, (y + .15) * cell_width
            x2, y2 = (x + .5) * cell_width, (y + .6) * cell_width
            x3, y3 = (x + .85) * cell_width, (y + .85) * cell_width

            self._symbolic_view.create_polygon(x1, y3, x2, y1, x3, y3, x2, y2, fill=colour)
        else:
            x1, y1 = (x + .15) * cell_width, (y + .15) * cell_width
            x2, y2 = (x + .4) * cell_width, (y + .5) * cell_width
            x3, y3 = (x + .85) * cell_width, (y + .85) * cell_width

            self._symbolic_view.create_polygon(x1, y3, x2, y2, x1, y1, x3, y2, fill=colour)

    def _poll_frame(self):
        """
        Main callback for UI rendering.
        Called at regular intervals.
        The method will ask the environment to provide a frame if available (not None).
        :return:
        """
        cell_width = CELL_WIDTH
        circle_radius = 10

        # are we done?
        if self._env.done and not self._episode_has_ended:
            self._on_episode_end()

        # build symbolic view
        board = None
        if self._env is not None:
            board, _ = self._env._internal_symbolic_builder.build(self._env)
        if board is not None:
            board = board.T
            self._symbolic_view.delete('all')  # Remove all previous items from Tkinter tracking
            width, height = board.shape
            for x in range(width):
                for y in range(height):
                    cell_contents = str.split(str(board[x][y]), '/')
                    for block in cell_contents:
                        if block == 'sand':
                            self._symbolic_view.create_rectangle(x * cell_width, y * cell_width,
                                                                 (x + 1) * cell_width, (y + 1) * cell_width,
                                                                 outline="black", fill="orange", tags="square")
                        elif block == 'grass':
                            self._symbolic_view.create_rectangle(x * cell_width, y * cell_width,
                                                                 (x + 1) * cell_width, (y + 1) * cell_width,
                                                                 outline="black", fill="lawn green", tags="square")
                        elif block == 'lapis_block':
                            self._symbolic_view.create_rectangle(x * cell_width, y * cell_width,
                                                                 (x + 1) * cell_width, (y + 1) * cell_width,
                                                                 outline="black", fill="black", tags="square")
                        elif block == ENV_TARGET_NAMES[0]:
                            self._symbolic_view.create_oval((x + .5) * cell_width - circle_radius,
                                                            (y + .5) * cell_width - circle_radius,
                                                            (x + .5) * cell_width + circle_radius,
                                                            (y + .5) * cell_width + circle_radius,
                                                            fill='pink')
                        elif block == self.name:
                            yaw = self._env._world_obs['Yaw'] % 360
                            self._draw_arrow(yaw, x, y, cell_width, 'red')
                        elif block == ENV_AGENT_NAMES[0]:
                            # Get yaw of other agent:
                            entities = self._env._world_obs[ENV_ENTITIES]
                            other_agent = list(
                                map(Entity.create, filter(lambda e: e['name'] == ENV_AGENT_NAMES[0], entities)))
                            if len(other_agent) == 1:
                                other_agent = other_agent.pop()
                                yaw = other_agent.yaw % 360
                                self._draw_arrow(yaw, x, y, cell_width, 'blue')

        # display the most recent frame
        frame = self._env.frame
        if frame is not None:
            from PIL import ImageTk
            self._first_person_view.image = ImageTk.PhotoImage(image=frame)
            self._first_person_view.configure(image=self._first_person_view.image)
            self._first_person_view.update()

        self._first_person_view.update()

        # process game state (e.g., has the episode started?)
        if self._episode_has_started and time.time() - self._episode_start_time < 3:
            if not hasattr(self, "_init_overlay") or not self._init_overlay:
                self._create_overlay()
            self._init_overlay.delete("all")
            self._init_overlay.create_rectangle(
                10, 10, 590, 290, fill="white", outline="red", width="5")
            self._init_overlay.create_text(
                300, 80, text="Get ready to catch the pig!",
                font=('Helvetica', '18'))
            self._init_overlay.create_text(
                300, 140, text=str(3 - int(time.time() - self._episode_start_time)),
                font=('Helvetica', '18'), fill="red")
            self._init_overlay.create_text(
                300, 220, width=460,
                text="How to play: \nUse the left/right arrow keys to turn, "
                     "forward/back to move. The pig is caught if it is "
                     "cornered without a free block to escape to.",
                font=('Helvetica', '14'), fill="black")
            self._root.update()

        elif self._episode_has_ended:

            if not hasattr(self, "_init_overlay") or not self._init_overlay:
                self._create_overlay()
            self._init_overlay.delete("all")
            self._init_overlay.create_rectangle(
                10, 10, 590, 290, fill="white", outline="red", width="5")
            self._init_overlay.create_text(
                300, 80, text='Finished episode %d of %d' % (self._episode, self._max_episodes),
                font=('Helvetica', '18'))
            self._init_overlay.create_text(
                300, 120, text='Score: %d' % sum(self._rewards),
                font=('Helvetica', '18'))
            if self._episode > 1:
                self._init_overlay.create_text(
                    300, 160, text='Average over %d episodes: %.2f' % (self._episode, np.mean(self._scores)),
                    font=('Helvetica', '18'))
            self._init_overlay.create_text(
                300, 220, width=360,
                text="Press RETURN to start the next episode, ESC to exit.",
                font=('Helvetica', '14'), fill="black")
            self._root.update()

        elif hasattr(self, "_init_overlay") and self._init_overlay:
            self._destroy_overlay()

        # trigger the next update
        self._root.after(self._tick, self._poll_frame)

    def _create_overlay(self):
        self._init_overlay = Canvas(self._root, borderwidth=0, highlightthickness=0, width=600, height=300, bg="gray")
        self._init_overlay.place(relx=0.5, rely=0.5, anchor='center')

    def _destroy_overlay(self):
        self._init_overlay.destroy()
        self._init_overlay = None

    def _on_key_pressed(self, e):
        """
        Main callback for keyboard events
        :param e:
        :return:
        """
        if e.keysym == 'Escape':
            self._quit()

        if e.keysym == 'Return' and self._episode_has_ended:

            if self._episode >= self._max_episodes:
                self._quit()

            # start the next episode
            self._action_taken = 0
            self._rewards = []
            self._episode += 1
            self._env.reset()

            self._on_episode_start()
            print('Starting episode %d' % self._episode)

        if self._episode_has_started and time.time() - self._episode_start_time >= 3:
            if e.keysym in self._keymap:
                mapped_action = self._keymap.index(e.keysym)

                _, reward, done = self._env.do(mapped_action)
                self._action_taken += 1
                self._rewards.append(reward)
                self._on_experiment_updated(mapped_action, reward, done)

    def _on_episode_start(self):
        self._episode_has_ended = False
        self._episode_has_started = True
        self._episode_start_time = time.time()
        self._on_experiment_updated(None, 0, self._env.done)

    def _on_episode_end(self):
        # do a turn to ensure we get the final reward and observation
        no_op_action = 0
        _, reward, done = self._env.do(no_op_action)
        self._action_taken += 1
        self._rewards.append(reward)
        self._on_experiment_updated(no_op_action, reward, done)

        # report scores
        self._scores.append(sum(self._rewards))
        self.visualize(self._episode, 'Reward', sum(self._rewards))

        # set flags to start a new episode
        self._episode_has_started = False
        self._episode_has_ended = True

    def _on_experiment_updated(self, action, reward, is_done):
        self._current_episode_lbl.config(text='Episode: %d' % self._episode)
        self._cum_reward_lbl.config(text='Score: %d' % sum(self._rewards))
        self._last_action_lbl.config(text='Previous action: %s' % action)
        self._action_done_lbl.config(text='Actions taken: {0}'.format(self._action_taken))
        self._action_remaining_lbl.config(text='Actions remaining: %d' % (self._max_actions - self._action_taken))
        self._first_person_view.update()

    def _quit(self):
        self._quit_event.set()
        self._root.quit()
        sys.exit()
