# generateTournamentLayouts.py
# ----------------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import sys, random

import mazeGenerator

"""
This is a helper file which generates the random seeds for the map
layouts for the nightly tournament.
"""

if __name__=="__main__":
  num = 9
  if len(sys.argv) > 1: # command line argument: number of maps to generate
    num = int(sys.argv[1])

  seedsfile = '../driver/SEEDS'
  with open(seedsfile,'w') as out:
    pass

  for i in range(num):
    seed = random.randint(0,99999999)
    layout = 'layouts/random%08dCapture.lay' % seed
    print 'Generating random layout in %s' % layout
    with open(layout, 'w') as out:
      maze = mazeGenerator.generateMaze(seed)
      out.write(maze)
      print maze

    with open(seedsfile, 'a') as out:
      out.write("%d\n"%seed)



