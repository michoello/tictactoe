import random
import network
import copy
import sys
import os.path

filename = sys.argv[1] if len(sys.argv) > 1 else 'network.json'
print 'filename: ', filename

net = network.Network([72, 36, 16, 2])
    
if os.path.isfile(filename):
    net = net.load(filename)

# ---------------------------------------------------------------------
class Game:
    def __init__(self):
        self.field = [[' '] * 6 for x in range(6)]
        self.history = []

    def show(self):
        print '   ' + ' '.join([' '+netstr(x)+' ' for x in range(0,6)])
        print '  -------------------------'
        for idx, line in enumerate(self.field):
            print idx,  '| ' + ' | '.join(line) + ' |'
            print '  -------------------------'

    def set(self, xy, val, field = None):
        if field == None: field = self.field 
        field[xy[0]][xy[1]] = val

    def get(self, xy):
        x, y = xy
        return self.field[x][y] if x in range(6) and y in range(6) else ' '

    def available(self):
        return [(y, x) for y, row in enumerate(self.field) for x, v in enumerate(row) if v == ' ']

    def winner(self, player):
        n = 4
        netstr = player.role * n
        for x in range(6):
            for y in range(6):
                hor = [(x, y + i) for i in range(n)]
                ver = [(x + i, y) for i in range(n)]
                dia1 = [(x + i, y + i) for i in range(n)]
                dia2 = [(x + i, y - i) for i in range(n)]

                hvd = [ ''.join([ self.get(xy) for xy in four ]) for four in [hor, ver, dia1, dia2] ]

                if netstr in hvd:
                    return True
        return False

    def draw(self):
        return not self.available()

    def play(self, players, show):
        winner = None 
        history = []
        gameOver = False
        while not gameOver: 
           for player in players:

               xy = player.ply(self)
               self.set(xy, player.role)
              
               history.append(self.plainarray())
 
               if show:
                   self.show()

               if self.winner(player): 
                   gameOver, winner = True, player
                   break

               if self.draw():
                   gameOver = True
                   break
  
        # game result to [0,1]
        y0 = 0.5 if winner == None else 1 if winner.role == 'X' else 0 
        sys.stdout.write(str(y0))
        sys.stdout.flush()

        for idx, position in enumerate(reversed(history)):
            net.train(position, [y0, 1.0/(idx+1)], 0.02, 20)

        return (winner, len(history))

    def plainarray(self, field = None):
        if field == None: field = self.field 
        return sum( map( lambda c : [int(c == 'X'), int(c == 'O')], [ c for row in field for c in row ]), [])

# ---------------------------------------------------------------------
class Player:
    def __init__(self, role):
        self.role = role

    def ply(self, game):
        raise NotImplementedError()


# ---------------------------------------------------------------------
class HumanPlayer(Player):
    def ply(self, game):
        while True:
            xy = tuple( [int(n) for n in raw_input(self.role + ' ply: ').split()] )

            if xy in game.available():
                return xy

            print '(', xy[0], ',', xy[1], ') are already set to "', game.get(xy), '"'


# ---------------------------------------------------------------------
class RandomPlayer(Player):
    def ply(self, game):
        return random.choice(game.available())

# ---------------------------------------------------------------------
class NeuralPlayer(Player):
    def ply(self, game):

        if random.random() > 0.8: # TODO: make it constructor parameter
            return random.choice(game.available())

        bestrate = 10
        bestdist = -10
        bestchoice = None
        for xy in game.available():
            field = copy.deepcopy(game.field)
            game.set(xy, self.role, field = field)
            (rate, dist) = net.predict(game.plainarray(field))

            rate = round(rate, 1)
            dist = int(1.0/dist)

            if rate < bestrate:  
                bestchoice, bestrate, bestdist = xy, rate, dist

            #if rate == bestrate and rate < 0.5 and dist < bestdist:
            #    bestchoice, bestrate, bestdist = xy, rate, dist

            #if rate == bestrate and rate > 0.5 and dist > bestdist:
            #    bestchoice, bestrate, bestdist = xy, rate, dist

        return bestchoice 

# ---------------------------------------------------------------------
class StubbornPlayer(Player):
    def __init__(self, role):
        Player.__init__(self, role)
        self.__ply__ = self.__ply__(0, 0, [(dx,dy) for dx in range(6) for dy in range(6)])

    def __ply__(self, x, y, steps):
        for (dx, dy) in steps:
            yield (x + dx, y + dy)

    def ply(self, game):
        while True:
            step = next(self.__ply__)
            if step in game.available():
                return step

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def competition(show = False, rounds = 100):
    results = {'X': [0, 0], 'O': [0, 0], ' ': [0, 0]}
    i = 0
    while i < rounds:
        #playerX, playerO = NeuralPlayer('X'), StubbornPlayer('O')
        #playerX, playerO = RandomPlayer('X'), StubbornPlayer('O')
        playerX, playerO = StubbornPlayer('X'), NeuralPlayer('O')

        i+=1
        tictactoe = Game()
        (winner, length) = tictactoe.play([ playerX, playerO ], show)

        role = ' ' if winner is None else winner.role
        results[role] = (results[role][0] + 1, results[role][1] + length)

    print
    for role in results:
        print role, ': ', results[role][0], " avg ", results[role][1]/(1 if results[role][0] == 0 else results[role][0])
    print

# ---------------------------------------------------------------------
def playwithme(roboPlayer = RandomPlayer('X')):    
    tictactoe = Game()
    winner = tictactoe.play([ roboPlayer, HumanPlayer('O') ], True)
    print 'Winner: ', winner.role


# ---------------------------------------------------------------------
#playwithme(StubbornPlayer('X'))

for i in range(100):
    competition(show=False, rounds = 50)
    net.save(filename + '.' + str(i))



