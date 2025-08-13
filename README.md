# Metachronal-Paddling-RL
PhD dissertation work on a reinforcement learning approach to studying metachronal paddling at low Reynolds number     
    
Description: This framework uses the reinforcement learning algorithm, tabular Q-learning, to let an artificial microswimmer agent learn to swim in a low Reynolds number fluid environment. The fluid environment is specified using MATLAB CFD codes built on the method of regularized Stokeslets. The swimmer self-learns to swim with a selected gait that we examine and compare to biological strokes used by real microswimmers.

Fluid Solvers Directory - contains MATLAB codes that perform fluid solves for paddlers with 2,3 and 4 sets of limbs in 2D or 3D

Reward Table Generators Directory - contains MATLAB scripts that fill out reward tables that get passed to Q-learning
