import random
import network
import copy
import sys
import os.path


mode = sys.argv[1] if len(sys.argv) > 1 else 'play'
filename = sys.argv[2] if len(sys.argv) > 2 else 'network.json'
print 'mode: ', mode
print 'filename: ', filename

#net = network.Network([72, 36, 16, 2])
net = network.Network([72, 36, 1])
    
if os.path.isfile(filename):
    net = net.load(filename)

# ---------------------------------------------------------------------
class Game:
    def __init__(self):
        self.field = [[' '] * 6 for x in range(6)]
        self.history = []

    def showScores(self, role):
        print '  -------------------------'
        for r in range(6):
            print r,  '|' + \
                '|'.join([ '{:3.0f}'.format(self.predictPly(role, (r,c)) * 1000) for c in range(6)]) + '|' 
#print '  -------------------------'



    def show(self):
        print '   ' + ' '.join([' '+ str(x) + ' ' for x in range(0,6)])
        print '  -------------------------'
        for idx, line in enumerate(self.field):
            print idx,  '| ' + ' | '.join(line) + ' |'

#print '  -------------------------'
        print 'Network opinion: ', self.predictPly()

        print 'Scores for O'
        self.showScores('O')

        print 'Scores for X'
        self.showScores('X')
            

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
        history = []
        while True: 
           for player in players:
               xy = player.ply(self)
               self.set(xy, player.role)
              
               history.append(self.plainarray())
 
               if show:
                   self.show()

               if self.winner(player): 
                   return (player, history)

               if self.draw():
                   return (None, history)
                   

    def plainarray(self, field = None):
        if field == None: field = self.field 
        return sum( map( lambda c : [int(c == 'X'), int(c == 'O')], [ c for row in field for c in row ]), [])


    def trainGame(self, winner, history):
        role = ' ' if winner is None else winner.role

        goal = 0.5 if role == ' ' else 1 if role == 'X' else 0 
        goalDelta = 0.025 * (0 if role == ' ' else -1 if role == 'X' else 1)

        for stepsToFinish, position in enumerate(reversed(history)):
           net.train(position, [goal + goalDelta * stepsToFinish], 0.02, 10)

    def predictPly(self, role = ' ', xy = None):
        field = self.field
        if xy != None:
            field = copy.deepcopy(self.field)
            self.set(xy, role, field = field)

        return net.predict(self.plainarray(field))[0]


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

    def __init__(self, role, randomRate = 0.0):
        Player.__init__(self, role)
        self.randomRate = randomRate

    def ply(self, game):
        if random.random() > 1 - self.randomRate: 
            return random.choice(game.available())

        bestrate = 0 if self.role == 'X' else 1
        bestchoice = None

        for xy in game.available():
            rate = game.predictPly(self.role, xy)

            if (self.role == 'X' and rate > bestrate) or \
               (self.role == 'O' and rate < bestrate):
                bestchoice, bestrate = xy, rate

        # esli vsyo ploho, pytaemsya navredit'
        if self.role == 'O' and bestrate > 0.5:
            for xy in game.available():
                rate = game.predictPly('X', xy)
                if rate > bestrate:
                    bestchoice, bestrate = xy, rate

        if self.role == 'X' and bestrate < 0.5:
            for xy in game.available():
                rate = game.predictPly('O', xy)
                if rate < bestrate:
                    bestchoice, bestrate = xy, rate

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

def pprint(s):
    sys.stdout.write(s)
    sys.stdout.flush()

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
def competition(show, rounds, epoch):
    playStub = (epoch % 2 == 1)
    pprint((' Test ' if playStub else 'Train ') + str(epoch) + ': ')

    results = {'X': [0, 0], 'O': [0, 0], ' ': [0, 0]}
    i = 0
    while i < rounds:
        if not playStub:
            playerX, playerO = NeuralPlayer('X',0.0), NeuralPlayer('O', 0.0)
        else:
            playerX, playerO = StubbornPlayer('X'), NeuralPlayer('O', 0.0)

        i+=1
        tictactoe = Game()
        (winner, history) = tictactoe.play([ playerX, playerO ], show)

        role = ' ' if winner is None else winner.role

        if not playStub: 
            tictactoe.trainGame(role, position, idx)

        results[role] = (results[role][0] + 1, results[role][1] + len(history))
        pprint(role)


    print
    for role, (cnt, leng) in results.iteritems():
        print role, ': ', cnt, " avg ", leng/(1 if cnt == 0 else cnt)
#print role, ': ', data[0], " avg ", data[1]/(1 if data[0] == 0 else data[0])
    print

# ---------------------------------------------------------------------
def playagame(playerX, playerO):    
    tictactoe = Game()
    winner, history = tictactoe.play([ playerX, playerO ], True)
    
    print 'Winner: ', winner.role
    
    tictactoe.trainGame(winner, history)
    rate = net.predict(tictactoe.plainarray())[0]
    print 'Network opinion after training: ', rate




# ---------------------------------------------------------------------
if mode == 'play':    
    while True:
        playagame(NeuralPlayer('X'), HumanPlayer('O'))
        #playagame(HumanPlayer('X'), NeuralPlayer('O', 0.0))

else:   
    for i in range(100):
        competition(show=False, rounds = 50, epoch = i)
        net.save(filename + '.' + str(i))

