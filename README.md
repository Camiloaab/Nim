# Nim

## An AI that teaches itself to play Nim through reinforcement learning.

<img src="resources/nim_output.png" width="1000">

In the game Nim, we begin with some number of piles, each with some number of objects. Players take turns: on a player’s turn, the player removes any non-negative number of objects from any one non-empty pile. Whoever removes the last object loses.

There’s some simple strategy we might imagine for this game: if there’s only one pile and three objects left in it, and it’s your turn, your best bet is to remove two of those objects, leaving your opponent with the third and final object to remove. But if there are more piles, the strategy gets considerably more complicated.

For this project, we built an AI to learn the strategy for this game through reinforcement learning. By playing against itself repeatedly and learning from experience, eventually our AI will learn which actions to take and which actions to avoid.

**Reinforcement Learning**

Reinforcement learning is an approach to machine learning, where after each action, the agent gets feedback in the form of reward or punishment (a positive or a negative numerical value).

The learning process starts by the environment providing a state to the agent. Then, the agent performs an action on the state. Based on this action, the environment will return a state and a reward to the agent, where the reward can be positive, making the behavior more likely in the future, or negative (i.e. punishment), making the behavior less likely in the future.

<img src="resources/reinforcement.png" width="600">

A “state” of the Nim game is just the current size of all of the piles. A state, for example, might be `[1, 1, 3, 5]`, representing the state with 1 object in pile 0, 1 object in pile 1, 3 objects in pile 2, and 5 objects in pile 3. An “action” in the Nim game will be a pair of integers `(i, j)`, representing the action of taking `j` objects from pile `i`. So the action `(3, 5)` represents the action “from pile 3, take away 5 objects.” Applying that action to the state `[1, 1, 3, 5]` would result in the new state `[1, 1, 3, 0]` (the same state, but with pile 3 now empty).

**Q-Learning**

Q-Learning is one model of reinforcement learning, where a function `Q(s, a)` outputs an estimate of the value of taking action `a` in state `s`. A way to train a reinforcement learning model is to give feedback not upon every move, but upon the end of the whole process. In Nim game, an untrained AI will play randomly, and it will be easy to win against it. To train the AI, it will start from playing a game randomly, and in the end get a reward of 1 for winning and -1 for losing, an action that results in the game continuing has an immediate reward of `0`. When it is trained on 10,000 games, for example, it is already smart enough to be hard to win against it.

The model starts with all estimated values equal to `0` `(Q(s,a) = 0 for all s, a)`. When an action is taken and a reward is received, the function does two things: 1) it estimates the value of `Q(s, a)` based on current reward and expected future rewards, and 2) updates `Q(s, a)` to take into account both the old estimate and the new estimate. This gives us an algorithm that is capable of improving upon its past knowledge without starting from scratch. The key formula for Q-learning is below. Every time we are in a state `s` and take an action `a`, we can update the Q-value `Q(s, a)` according to:

`Q(s, a) <- Q(s, a) + α(new value estimate - old value estimate)`

In the above formula, alpha `α` is the learning rate (how much we value new information compared to information we already have). The new value estimate represents the sum of the reward received for the current action and the estimate of all the future rewards that the player will receive. The old value estimate is just the existing value for `Q(s, a)`. By applying this formula every time our AI takes a new action, over time our AI will start to learn which actions are better in any state. The updated value of `Q(s, a)` is equal to the previous value of `Q(s, a)` in addition to some updating value. This value is determined as the difference between the new value and the old value, multiplied by `alpha`, a learning coefficient. When `alpha = 1` the new estimate simply overwrites the old one. When `alpha = 0`, the estimated value is never updated. By raising and lowering `alpha`, we can determine how fast previous knowledge is being updated by new estimates.

The new value estimate can be expressed as a sum of the reward and the future reward estimate. To get the future reward estimate, we consider the new state that we got after taking the last action, and add the estimate of the action in this new state that will bring to the highest reward. This way, we estimate the utility of making action `a` in state `s` not only by the reward it received, but also by the expected utility of the next step. The value of the future reward estimate can sometimes appear with a coefficient gamma `ɣ` that controls how much future rewards are valued. We end up with the following equation:

`Q(s, a) <- Q(s, a) + α((reward + ɣ(max a' Q(s', a'))) - Q(s, a))`

**Explore vs. Exploit**

A Greedy Decision-Making algorithm completely discounts the future estimated rewards, instead always choosing the action `a` in current state `s` that has the highest `Q(s, a)`. A greedy algorithm always exploits, taking the actions that are already established to bring to good outcomes. However, it will always follow the same path to the solution, never finding a better path. 

Exploration, on the other hand, means that the algorithm may use a previously unexplored route on its way to the target, allowing it to discover more efficient solutions along the way. 

To implement the concept of exploration and exploitation, we can use the epsilon `ε` greedy algorithm. In this type of algorithm, we set `ε` equal to how often we want to move randomly. With probability `1-ε`, the algorithm chooses the best move (exploitation). With probability `ε`, the algorithm chooses a random move (exploration).

## Implementation

In `nim.py`, there are two classes defined: `Nim` and `NimAI`, along with two functions: `train` and `play`.

The `Nim` class defines how a Nim game is played. In the `__init__` function, every Nim game keeps track of a list of piles, a current player (`0` or `1`), and the winner of the game (if one exists). The `available_actions` function returns a set of all the available actions in a state. For example, `Nim.available_actions([2, 1, 0, 0])` returns the set `{(0, 1), (1, 1), (0, 2)}`, since the three possible actions are to take either 1 or 2 objects from pile `0`, or to take 1 object from pile `1`.

The remaining functions are used to define the gameplay: the `other_player` function determines who the opponent of a given player is, `switch_player` changes the current player to the opposing player, and `move` performs an action on the current state and switches the current player to the opposing player.

The `NimAI` class defines our AI that will learn to play Nim. In the `__init__` function, we start with an empty `self.q` dictionary, this will keep track of all of the current Q-values learned by our AI by mapping `(state, action)` pairs to a numerical value. As an implementation detail, though we usually represent state as a list, since lists can’t be used as Python dictionary keys, we instead use a tuple version of the state when getting or setting values in `self.q`.

For example, if we wanted to set the Q-value of the state `[0, 0, 0, 2]` and the action `(3, 2)` to `-1`, we would write something like:

`self.q[(0, 0, 0, 2), (3, 2)] = -1`

Every `NimAI` object has an `alpha` and `epsilon` value that will be used for Q-learning and for action selection, respectively.

The `update` function takes as input state `old_state`, an action taken in that state `action`, the resulting state after performing that action `new_state`, and an immediate reward for taking that action `reward`. The function then performs Q-learning by first getting the current Q-value for the `state` and `action` (by calling `get_q_value`), determining the best possible future rewards (by calling `best_future_reward`), and then using both of those values to update the Q-value (by calling `update_q_value`).

The `choose_action` function selects an action to take in a given state (either greedily, or using the epsilon-greedy algorithm).

The `Nim` and `NimAI` classes are ultimately used in the `train` and `play` functions. The `train` function trains an AI by running `n` simulated games against itself, returning the fully trained AI. The `play` function accepts a trained AI as input, and lets a human player play a game of Nim against the AI.

## Resources

* [Learning - Lecture 4 - CS50's Introduction to Artificial Intelligence with Python 2020][cs50 lecture]

## Usage

**To train an AI and play Nim against it:** 

* Inside the `nim` directory: `python play.py`

## Acknowledgements

Project and images from the course [CS50's Introduction to Artificial Intelligence with Python 2020][cs50 ai] from HarvardX.

## Credits

[*Luis Sanchez*][linkedin] 2021.

[cs50 lecture]: https://youtu.be/E4M_IQG0d9g?t=4198
[linkedin]: https://www.linkedin.com/in/luis-sanchez-13bb3b189/
[cs50 ai]: https://cs50.harvard.edu/ai/2020/
