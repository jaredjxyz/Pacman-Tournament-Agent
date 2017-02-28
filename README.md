# Team Organic Stupidity
### Jared Jensen and Chris Gradwohl, University of California, Santa Cruz


## Contest: Pacman Capture the Flag

> <center>![](capture_the_flag.png)</center>
>
> <center>Enough of defense,  
> Onto enemy terrain.  
> Capture all their food!</center>

### Introduction

The final project involves a multi-player capture-the-flag variant of Pacman, where agents control both Pacman and ghosts in coordinated team-based strategies. Your team will try to eat the food on the far side of the map, while defending the food on your home side.

**On 5PM Wednesday March 01** , we will run an initial test tournament with everyone's dummy submissions (more details below). By this date, you must have selected team names and members. We'll be having round-robin style tournaments between student submissions and announce nightly match results starting at **10PM Friday March 03** . The final tournament will be run at **10PM Monday March 13** . 2-3 page project writeups should be turned in by **5PM Friday March 17** .

We will evaluate your submissions based on a short written report (2-3 pages) on your modeling of the problem and agent design, as well as your performance against your classmates in tournament play.

The remainder of this file describes the following important details: code files, rules of the capture-the-flag game, submission instructions, tournament logistics, grading and scoring rubric, and tips on getting started.

Please read the instructions very carefully and thoroughly and make note of the important dates at the bottom of the page.

<table border="0" cellpadding="10">

<tbody>

<tr>

<td>**File to submit:**</td>

</tr>

<tr>

<td>`[myTeam.py](docs/myTeam.html)`</td>

<td>This is where you define your own agents for inclusion in the nightly tournament. (This is the only file that you submit.)</td>

</tr>

<tr>

<td>**Key files to read:**</td>

</tr>

<tr>

<td>`[capture.py](docs/capture.html)`</td>

<td>The main file that runs games locally. This file also describes the new capture the flag GameState type and rules.</td>

</tr>

<tr>

<td>`[captureAgents.py](docs/captureAgents.html)`</td>

<td>Specification and helper methods for capture agents.</td>

</tr>

<tr>

<td>`[baselineTeam.py](docs/baselineTeam.html)`</td>

<td>Example code that defines two very basic reflex agents, to help you get started.</td>

</tr>

<tr>

<td>`[myTeam.py](docs/myTeam.html)`</td>

<td>This is where you define your own agents for inclusion in the nightly tournament. (This is the only file that you submit.) Currently, there is a Dummy Agent defined to help you get started.</td>

</tr>

<tr>

<th colspan="2" align="left">**Supporting files (do not modify):**</th>

</tr>

<tr>

<td>`[game.py](docs/game.html)`</td>

<td>The logic behind how the Pacman world works. This file describes several supporting types like AgentState, Agent, Direction, and Grid.</td>

</tr>

<tr>

<td>`[util.py](docs/util.html)`</td>

<td>Useful data structures for implementing search algorithms.</td>

</tr>

<tr>

<td>`[distanceCalculator.py](docs/distanceCalculator.html)`</td>

<td>Computes shortest paths between all maze positions.</td>

</tr>

<tr>

<td>`[graphicsDisplay.py](docs/graphicsDisplay.html)`</td>

<td>Graphics for Pacman</td>

</tr>

<tr>

<td>`[graphicsUtils.py](docs/graphicsUtils.html)`</td>

<td>Support for Pacman graphics</td>

</tr>

<tr>

<td>`[textDisplay.py](docs/textDisplay.html)`</td>

<td>ASCII graphics for Pacman</td>

</tr>

<tr>

<td>`[keyboardAgents.py](docs/keyboardAgents.html)`</td>

<td>Keyboard interfaces to control Pacman</td>

</tr>

<tr>

<td>`[layout.py](docs/layout.html)`</td>

<td>Code for reading layout files and storing their contents</td>

</tr>

</tbody>

</table>

**Academic Dishonesty:** While we won't grade the source code of contest submissions directly, we still expect you not to falsely represent your work. _Please_ don't let us down.

### Rules of Pacman Capture the Flag

**Layout:** The Pacman map is now divided into two halves: blue (right) and red (left). Red agents (which all have even indices) must defend the red food while trying to eat the blue food. When on the red side, a red agent is a ghost. When crossing into enemy territory, the agent becomes a Pacman.

**Scoring:** When a Pacman eats a food dot, the food is permanently removed and one point is scored for that Pacman's team. Red team scores are positive, while Blue team scores are negative.

**Eating Pacman:** When a Pacman is eaten by an opposing ghost, the Pacman returns to its starting position (as a ghost). No points are awarded for eating an opponent.

**Power capsules:** If Pacman eats a power capsule, agents on the opposing team become "scared" for the next 40 moves, or until they are eaten and respawn, whichever comes sooner. Agents that are "scared" are susceptible while in the form of ghosts (i.e. while on their own team's side) to being eaten by Pacman. Specifically, if Pacman collides with a "scared" ghost, Pacman is unaffected and the ghost respawns at its starting position (no longer in the "scared" state).

**Observations:** Agents can only observe an opponent's configuration (position and direction) if they or their teammate is within 5 squares (Manhattan distance). In addition, an agent always gets a noisy distance reading for each agent on the board, which can be used to approximately locate unobserved opponents.

**Winning:** A game ends when one team eats all but two of the opponents' dots. Games are also limited to 1200 agent moves (300 moves per each of the four agents). If this move limit is reached, whichever team has eaten the most food wins. If the score is zero (i.e., tied) this is recorded as a tie game.

**Computation Time:** We will run your submissions on a VM server. Each move which does not return within one second will incur a warning. After three warnings, or any single move taking more than 3 seconds, the game is forfeit. There will be an initial start-up allowance of 15 seconds (use the `registerInitialState` function). If you agent times out or otherwise throws an exception, an error message will be present in the log files, which you can download from the results page (see below).

### Evaluation

The contest will count as your final project, worth 25 points. 20 of these points will be the result of a written report you submit with your agent describing your approach. The remaining 5 points will be awarded based on your performance in the final contest.

The written report should be 2-3 pages (no more). Through this report we expect you to demonstrate your ability to constructively solve AI problems by identifying:

*   the fundamental problems you are trying to solve
*   how you modeled these problems
*   your representations of the problems
*   the computational strategy used to solve each problem
*   algorithmic choices you made in your implementation
*   any obstacles you encountered while solving the problem
*   evaluation of your agent
*   lessons learned during the project

The performance-based portion of your grade will be awarded as follows:

*   2 points for a solution that runs
*   3 points for being in the top 75%
*   4 points for being in the top 40%
*   5 points for being in the top 20%

How we compute the percentiles based on the ranking of the teams is described below in the Contest Details.

### Submission Instructions

**VERY IMPORTANT! PLEASE READ THOROUGHLY.**

To enter into the nightly tournaments, your team's agents and all relevant functions must be defined in `[myTeam.py](docs/myTeam.html)`. Due to the way the tournaments are run, your code must not rely on any additional files that we have not provided (The submission system may allow you to submit additional files, but the contest framework will not include them when your code is run in the tournament). You may not modify the code we provide.

Every team must have a unique name, consisting only of ASCII letters and digits. (Any other characters, including whitespace, will be ignored.) By Tuesday, February 28, you must fill in your team name and list all team members in the form [here](https://bit.do/teamNames) . Please access this document using your UCSC account as this is intended to be shared with only with valid UCSC accounts. As shown in the Google doc, you will state your chosen team name and beneath it list all team members.

In every submission to the autograder (linked below), you must include a file `name.txt` in which you will write only your unique team name. Do not include other extraneous text in this file. Only your team name will be displayed to the rest of the class.

To submit, use our submit server [here](http://cmps140.soe.ucsc.edu/p5/submit.html). Submit a zip file named solution. Zip all necessary files with the command:

<pre><small>zip solution.zip myTeam.py name.txt</small></pre>

Include additional files in the submission if you need to. Since there isn't a "correct" solution to this assignment, the submit server test will report whether your agent beat the provided `baseLineTeam` agent within the contest rules.

For the first checkpoint on March 1, you can submit the Dummy Agent given to you or the Baseline Agent. This is just a test run to make sure the tournaments run well and will not be used in the final score for your team.

For your final submission, include a file named "[your team name].pdf" that contains your write-up. Please make sure that this document contains the names of all members of your team clearly stated at the top.

### Getting Started

By default, you can run a game with the simple `baselineTeam` that the staff has provided:

<pre>python capture.py</pre>

A wealth of options are available to you:

<pre>python capture.py --help</pre>

There are four slots for agents, where agents 0 and 2 are always on the red team, and 1 and 3 are on the blue team. Agents are created by agent factories (one for Red, one for Blue). See the section on designing agents for a description of the agents invoked above. The only team that we provide is the `baselineTeam`. It is chosen by default as both the red and blue team, but as an example of how to choose teams:

<pre>python capture.py -r baselineTeam -b baselineTeam</pre>

which specifies that the red team `-r` and the blue team `-b` are both created from `[baselineTeam.py](docs/baselineTeam.html)`. To control one of the four agents with the keyboard, pass the appropriate option:

<pre>python capture.py --keys0</pre>

The arrow keys control your character, which will change from ghost to Pacman when crossing the center line.

### Layouts

By default, all games are run on the `defaultcapture` layout. To test your agent on other layouts, use the `-l` option. In particular, you can generate random layouts by specifying `RANDOM[seed]`. For example, `-l RANDOM13` will use a map randomly generated with seed 13.

### Game Types

You can play the game in two ways: local games, and nightly tournaments.

Local games (described above) allow you to test your agents against the baseline teams we provide and are intended for use in development.

### <a name="tournaments">Official Tournaments</a>

<a name="tournaments">The round-robin contests will be run using nightly automated tournaments on the course server, with the final tournament deciding the final contest outcomes. See the submission instructions for details of how to enter a team into the tournaments. Tournaments are run everyday at approximately 10pm and include all teams that have been submitted (either earlier in the day or on a previous day) as of the start of the tournament. Currently, each team plays every other team 3 times during the regular nightly matches and 5 times in the final tournament (to reduce randomness), but this may change.</a>

<a name="tournaments">The layouts used in the tournament will be drawn from both the default layouts included in the zip file as well as randomly generated layouts each night. All layouts are symmetric, and the team that moves first is randomly chosen. The results are available at</a> [http://cmps140.soe.ucsc.edu/tournament/2017-03-dd/results.html](http://cmps140.soe.ucsc.edu/tournament/2017777777-03-dd/results.html) after the tournament completes each night - you can view overall rankings and scores for each match. At the URL, substitute dd with the day of March of the tournament results you want to access. You can also download replays, the layouts used, and the stdout / stderr logs for each agent.

### Designing Agents

Unlike project 2, an agent now has the more complex job of trading off offense versus defense and effectively functioning as both a ghost and a Pacman in a team setting. Furthermore, the limited information provided to your agent will likely necessitate some probabilistic tracking (like we learned in the M5 module). Finally, the added time limit of computation introduces new challenges.

**Baseline Team:** To kickstart your agent design, we have provided you with a team of two baseline agents, defined in `[baselineTeam.py](docs/baselineTeam.html)`. They are both quite bad. The `OffensiveReflexAgent` moves toward the closest food on the opposing side. The `DefensiveReflexAgent` wanders around on its own side and tries to chase down invaders it happens to see.

**File naming:** For the purpose of testing or running games locally, you can define a team of agents in any arbitrarily-named python file. When submitting to the nightly tournament, however, you must define your agents in `[myTeam.py](docs/myTeam.html)` (and you must also create a `name.txt` file that specifies your team name).

**Interface:** The `GameState` in `[capture.py](docs/capture.html)` should look familiar, but contains new methods like `getRedFood`, which gets a grid of food on the red side (note that the grid is the size of the board, but is only true for cells on the red side with food). Also, note that you can list a team's indices with `getRedTeamIndices`, or test membership with `isOnRedTeam`.

Finally, you can access the list of noisy distance observations via `getAgentDistances`. These distances are within 6 of the truth, and the noise is chosen uniformly at random from the range [-6, 6] (e.g., if the true distance is 6, then each of {0, 1, ..., 12} is chosen with probability 1/13). You can get the likelihood of a noisy reading using `getDistanceProb`.

**Distance Calculation:** To facilitate agent development, we provide code in `[distanceCalculator.py](docs/distanceCalculator.html)` to supply shortest path maze distances.

To get started designing your own agent, we recommend subclassing the `CaptureAgent` class. This provides access to several convenience methods. Some useful methods are:

<pre>  def getFood(self, gameState):
    """
    Returns the food you're meant to eat. This is in the form
    of a matrix where m[x][y]=true if there is food you can
    eat (based on your team) in that square.
    """
  def getFoodYouAreDefending(self, gameState):
    """
    Returns the food you're meant to protect (i.e., that your
    opponent is supposed to eat). This is in the form of a
    matrix where m[x][y]=true if there is food at (x,y) that
    your opponent can eat.
    """
  def getOpponents(self, gameState):
    """
    Returns agent indices of your opponents. This is the list
    of the numbers of the agents (e.g., red might be "1,3,5")
    """
  def getTeam(self, gameState):
    """
    Returns agent indices of your team. This is the list of
    the numbers of the agents (e.g., red might be "1,3,5")
    """
  def getScore(self, gameState):
    """
    Returns how much you are beating the other team by in the
    form of a number that is the difference between your score
    and the opponents score. This number is negative if you're
    losing.
    """
  def getMazeDistance(self, pos1, pos2):
    """
    Returns the distance between two points; These are calculated using the provided
    distancer object.
    If distancer.getMazeDistances() has been called, then maze distances are available.
    Otherwise, this just returns Manhattan distance.
    """
  def getPreviousObservation(self):
    """
    Returns the GameState object corresponding to the last
    state this agent saw (the observed state of the game last
    time this agent moved - this may not include all of your
    opponent's agent locations exactly).
    """
  def getCurrentObservation(self):
    """
    Returns the GameState object corresponding this agent's
    current observation (the observed state of the game - this
    may not include all of your opponent's agent locations
    exactly).
    """
  def debugDraw(self, cells, color, clear=False):
    """
    Draws a colored box on each of the cells you specify. If clear is True,
    will clear all old drawings before drawing on the specified cells.
    This is useful for debugging the locations that your code works with.
    color: list of RGB values between 0 and 1 (i.e. [1,0,0] for red)
    cells: list of game positions to draw on  (i.e. [(20,5), (3,22)])
    """
</pre>

**Restrictions:** You are free to design any agent you want. However, you will need to respect the provided APIs if you want to participate in the tournaments. Agents which compute during the opponent's turn will be disqualified. In particular, any form of multi-threading is disallowed, because we have found it very hard to ensure that no computation takes place on the opponent's turn.

**Warning:** If one of your agents produces any stdout/stderr output during its games in the nightly tournaments, that output will be included in the contest results posted on the website. Additionally, in some cases a stack trace may be shown among this output in the event that one of your agents throws an exception. You should design your code in such a way that this does not expose any information that you wish to keep confidential.

### Contest Details

**Teams:** We highly encourage you to work in teams of 2 people (no more than 2).

**Prizes:** The final rankings used in determining the 5 points portion of your grade will be based on the number of points received in **the final** round-robin tournament, where a win is worth 4 points, a tie is worth 1 point, and losses are worth 0 (Ties are not worth very much to discourage stalemates).To be included in the nightly tournaments, your submission must be in by 10:00pm that night.

Extra credit will be awarded according to the nightly tournament rankings as follows:

We will award extra credit points to the teams that submit *any* solution at the first checkpoint (submitted by 3/1). You may submit the baseline agent or dummy agent.

Winners in the mid-project checkpoint contest (run on 3/8) will receive points as follows: 3 points for 1st place, 2 points for 2nd place, and 1 point for third place.

Winners in the final contest (run on 3/17) will receive points as follows: 3 points for 1st place, 2 points for 2nd place, and 1 point for third place.

**Important dates:**

<table border="0" cellspacing="5" cellpadding="5">

<tbody>

<tr>

<td>Thursday</td>

<td>02/23/2017</td>

<td>Contest announced and posted</td>

</tr>

<tr>

<td>Wednesday</td>

<td>03/01/2017</td>

<td>First Checkpoint: test run of the tournament. Any team that has submitted something (even only the teams.py file with the random agent unchanged) will get aforemention extra credit points.</td>

</tr>

<tr>

<td>Friday</td>

<td>03/03/2017</td>

<td>Nightly tournaments start running at 10PM daily</td>

</tr>

<tr>

<td>Wednesday</td>

<td>03/08/2017</td>

<td>Midway Checkpoint. Tournament winners so far receive extra credit.</td>

</tr>

<tr>

<td>Monday</td>

<td>03/13/2017</td>

<td>Last tournament is run.</td>

</tr>

<tr>

<td>Friday</td>

<td>03/17/2017</td>

<td>Project report due (by 5pm)</td>

</tr>

</tbody>

</table>

### Acknowledgements

Thanks to Barak Michener and Ed Karuna for providing improved graphics and debugging help.

![](capture_the_flag2.png)

Have fun! Please bring our attention to any problems you discover.
