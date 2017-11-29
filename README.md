# DQNAgent-TheMalmoCollaborativeAIChallenge
Final project of my BSc in Computer Science where I developed an AI for the Microsoft's The Malmo Collaborative AI Challenge. The agent uses a variation adapted to the malmo platfor of DeepMind's Deep Reinforcement Learning algorithm.

*WARNING*

To test the game you must have installed Project Malmo, link here:

https://github.com/Microsoft/malmo#getting-started


Also, to thest the agent I created you must have installed the Keras library for python with all its dependencies, as it is used to create and manage the artificial neural network:

https://keras.io/

*HOW TO USE AND THEST THE AGENT*

1.- Have Project Malmo & Keras correctly installed
2.- Download the project of the pig chase game from https://github.com/Microsoft/malmo-challenge
3.- Add the files of the folder agents-pig-chase of this repository to the folder ai_challenge/pig_chase of the malmo-challenge             repository 
4.- Start two instances of the Malmo Client on ports 10000 and 10001
5.- cd malmo-challenge/ai_challenge/pig_chase
6.- python luarsu_evaluation.py
7.- Try the agent and make your own experiments with it!

*CHECK/CHANGE THE IMPORTANT CODE*

The scripts/files where the DQN agent is implemented are:

-agent.py (The class for the DQN agent is the FocusedAgent class)
-environment.py (Where the main loop of the algorithm and the game are executed)
-luarsu_evaluation.py (The main code that starts the game)

All the code it's commented to make it as clear as possible and make the algorithm more understandable.

*EXPLANATION*


This project was developed during my last semester of my BSc in Computer Engineering as my Bachelor's final project at Universidad Politecnica de Valencia and was part of the Microsoft's contest The Malmo Collaborative AI Challenge. This aim of this contest was to encourage research in collaboration between intelligent agents. To do this, the proposed the following challenge: To develop an AI that learnt to collaborate with another one without involving any communication between them to obtain the highest score possible in the mini game Pig Chase.

This mini game was implemented in Minecraft using Project Malmo; an AI experimentation platform built on top of Minecraft. Pig Chase is inspired by the variant of the stag hunt; a classical game theoretic game formulation that captures conflicts between collaboration and individual safety.

In the game the agent has two options: Try to collaborate with the other agent to catch the pig or give up and go to the exit. To catch the pig both agents have to corner the pig somewhere in the map so it can't move anywhere, and it provides a +25 reward to both of the agents. Giving up and going to the exit provides a +5 points reward to the agent that reaches it first. Each action performed in the game provides a negative reward of -1 points.

With this dilemma where the agent doesn't know if the other agent is going to collaborate or not, an interesting situation to study different AI behaviours is presented.

As the aim of the challenge was to provide more information about Ai collaboration, the main objective for my proposed solution was to be as general as possible, so it could be adapted easily to other games or dilemmas and, therefore, the information obtained from the analysis of this agent and its behaviour provided valuable data to the AI collaboration research and not only for this concrete game. Also, this solution was intended to be as similar as possible to the human cognitive and learning process.

With this objective in mind, I decided to use the DQN algorithm to study its behaviour and if it can learn not only to play the game but also to collaborate. This algorithm uses the data that the agent perceives to make a decision in each step of the game using an artificial neural network. After an action is performed and the reward/punishment for it is received, the agent uses this information to train the neural network, so it learns and takes better decisions at each step of the game after many iterations.

To do this, I adapted the algorithm for it to work in Project Malmo. The information provided to the agent's neural network at each step of the game was the position of the agent, the position of the pig and the position of the other agent. With this data, the actions taken, and the rewards obtained the agent adjusted the weights of the neural network so it made better predictions of which was the best action to take at each step. 

After created the agent, I trained it during 50,000 games and I made an analysis with the data I retrieved from the training and different tests. The final results showed that after 30,000 games the agent started to improve its results considerably and it also reached a solution (catching the pig or going to the exit) more often as the training continued. This, along other data retrieved, proved that the agent learnt to play the game (it improves its score with training) and to collaborate (the more games it played, the collaborative solution of catching the pig with the other agent was achieved) only using the perceptions it received from the environment.
