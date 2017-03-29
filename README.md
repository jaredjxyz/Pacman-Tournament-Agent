### Introduction

This was the final project for my Artificial Intelligence class at UCSC. Because I had learned some of this beforehand and knew some information in other areas that put me at an advantage, I put a lot of work on into this project. Please see the [write-up on my website](http://jaredj.xyz/projects/pacman_tournament_agent/). This project ended up getting the highest win/lose ratio in the class!

### Rules of Pacman Capture the Flag

**Layout:** The Pacman map is now divided into two halves: blue (right) and red (left). Red agents (which all have even indices) must defend the red food while trying to eat the blue food. When on the red side, a red agent is a ghost. When crossing into enemy territory, the agent becomes a Pacman.

**Scoring:** When a Pacman eats a food dot, the food is permanently removed and one point is scored for that Pacman's team. Red team scores are positive, while Blue team scores are negative.

**Eating Pacman:** When a Pacman is eaten by an opposing ghost, the Pacman returns to its starting position (as a ghost). No points are awarded for eating an opponent.

**Power capsules:** If Pacman eats a power capsule, agents on the opposing team become "scared" for the next 40 moves, or until they are eaten and respawn, whichever comes sooner. Agents that are "scared" are susceptible while in the form of ghosts (i.e. while on their own team's side) to being eaten by Pacman. Specifically, if Pacman collides with a "scared" ghost, Pacman is unaffected and the ghost respawns at its starting position (no longer in the "scared" state).

**Observations:** Agents can only observe an opponent's configuration (position and direction) if they or their teammate is within 5 squares (Manhattan distance). In addition, an agent always gets a noisy distance reading for each agent on the board, which can be used to approximately locate unobserved opponents.

**Winning:** A game ends when one team eats all but two of the opponents' dots. Games are also limited to 1200 agent moves (300 moves per each of the four agents). If this move limit is reached, whichever team has eaten the most food wins. If the score is zero (i.e., tied) this is recorded as a tie game.

### Getting Started

By default, you can run a game with the simple `baselineTeam` that the staff has provided:

`python capture.py`: Run the game with two baseline teams facing each other

`python capture.py -r myTeam -b baselineTeam`: Run the game with my custom team against the baseline team

`python capture.py --help`: See other options available to you

### Layouts

By default, all games are run on the `defaultcapture` layout. To test your agent on other layouts, use the `-l` option. In particular, you can generate random layouts by specifying `RANDOM[seed]`. For example, `-l RANDOM13` will use a map randomly generated with seed 13.

### Files to look at:

`myTeam.py`: All of the custom code is inside of there. This is where I define the defence and offence agents.
`captureAgent.py`: This is the base class of the custom agent. It gives a lot of resources for finding things on the map.
`capture.py`: This has things like the game state and other methods to find information about the map


