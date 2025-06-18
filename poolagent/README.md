# poolagent

This is the source code for the pool physics environment, making use of PoolTool. Several scripts in the main directory make use of this source code, and none of these files are to run directly.

## utils.py

This is where many common classes and methods have been put, mainly the State class and heuristic executing methods. Events are also defined here, its a bit of a mess.

## deps/

This folder contains the few dependencies, set here as there are minor augmentations. Most notably adding image support for openai models in DSPy.

## dspy_definition.py

This is collection of DSPy signatures, used throughout the DSPy-based LLM agents, i.e. in 'react_agents.py'.

## pool.py 

This is a wrapper around the PoolTool instance, and establishes the rules of the pool game environment. PoolGame allows for a head to head game, where control of whos taking shots switches when a foul is made. It's very important to note that PoolTool does not like being run in parallel, so only one Pool instance is used at once.

## pool_solver.py

This is the code to perform the black box optimsation to turn lists of events into shot parameters. Currently there is 
    - Bayesian Optimisation
    - Simulated Annealing 
    - Brute Force 

## llm.py

A wrapper around several LLM APIs and backends. Was commonly used before I switched to using DSPy for most LLM use cases. Likely not needed anymore.

## mcts.py

MCTS logic for making shots in the pool environment. Used by the MCTS league agents and the MCTS data generation.