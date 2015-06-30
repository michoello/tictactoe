import random
import network

net = network.Network([72, 36, 16, 2])

# ---------------------------------------------------------------------
class Game:
    def __init__(self):
        self.field = [[' '] * 6 for x in range(6)]
        self.history = []

    def show(self):
        print '   ' + ' '.join([' '+str(x)+' ' for x in range(0,6)])
        print '  -------------------------'
        for idx, line in enumerate(self.field):
            print idx,  '| ' + ' | '.join(line) + ' |'
            print '  -------------------------'

    def set(self, xy, val):
        self.field[xy[0]][xy[1]] = val

    def get(self, xy):
        x, y = xy
        return self.field[x][y] if x in range(6) and y in range(6) else ' '

    def available(self):
        return [(y, x) for y, row in enumerate(self.field) for x, v in enumerate(row) if v == ' ']

    def winner(self, player):
        n = 4
        str = player.role * n
        for x in range(6):
            for y in range(6):
                hor = [(x, y + i) for i in range(n)]
                ver = [(x + i, y) for i in range(n)]
                dia1 = [(x + i, y + i) for i in range(n)]
                dia2 = [(x + i, y - i) for i in range(n)]

                hvd = [ ''.join([ self.get(xy) for xy in four ]) for four in [hor, ver, dia1, dia2] ]

                if str in hvd:
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
        
        if winner != None:
           y0 = 1 if winner.role == 'X' else 0
           for idx, position in enumerate(reversed(history)):
               net.train(position, [y0, 1.0/(idx+1)], 0.5, 10/(idx+1))
               # print
               # print 'Position:', idx + 1, ' -> ', position
               # print 'Predict: ', net.predict(position)

        return winner

    def plainarray(self):
        return sum( map( lambda c : [int(c == 'X'), int(c == 'O')], [ c for row in self.field for c in row ]), [])

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
    results = {'X': 0, 'O': 0, ' ': 0}
    i = 0
    while i < rounds:
        playerX, playerO = StubbornPlayer('X'), RandomPlayer('O')
        #playerX, playerO = RandomPlayer('X'), StubbornPlayer('O')

        i+=1
        tictactoe = Game()
        winner = tictactoe.play([ playerX, playerO ], show)
        if winner is None:
            results[' '] = results[' '] + 1
        else:
            results[winner.role] = results[winner.role] + 1

    print 'X: ', results['X']
    print 'O: ', results['O']
    print ' : ', results[' ']

# ---------------------------------------------------------------------
def playwithme(roboPlayer = RandomPlayer('X')):    
    tictactoe = Game()
    winner = tictactoe.play([ roboPlayer, HumanPlayer('O') ], True)
    print 'Winner: ', winner.role


# ---------------------------------------------------------------------
#playwithme(StubbornPlayer('X'))

competition(show=True, rounds = 1)



