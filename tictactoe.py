import random

# ---------------------------------------------------------------------
class Game:
    def __init__(self):
        self.field = [[' '] * 6 for x in xrange(6)]

    def show(self):
        print '   ' + ' '.join([' '+str(x)+' ' for x in xrange(0,6)])
        print '  -------------------------'
        for idx, line in enumerate(self.field):
            print idx,  '| ' + ' | '.join(line) + ' |'
            print '  -------------------------'

    def set(self, xy, val):
        self.field[xy[0]][xy[1]] = val

    def get(self, xy):
        x, y = xy
        return self.field[x][y] if 0 <= x < 6 and 0 <= y < 6 else ' '

    def available(self):
        return [(y, x) for y, row in enumerate(self.field) for x, v in enumerate(row) if v == ' ']

    def winner(self, player):
        n = 4
        str = player.role * n
        for x in range(0, 6):
            for y in range(0, 6):
                hor = [(x, y + i) for i in range(0, n)]
                ver = [(x + i, y) for i in range(0, n)]
                dia1 = [(x + i, y + i) for i in range(0, n)]
                dia2 = [(x + i, y - i) for i in range(0, n)]

                hvd = [ ''.join([ self.get(xy) for xy in four ]) for four in [hor, ver, dia1, dia2] ]

                if str in hvd:
                    return True
        return False

    def draw(self):
        return not self.available()

    def play(self, players, show):
        while True: 
           for player in players:
               xy = player.ply(self)
               self.set(xy, player.role)
 
               if show: self.show()

               if self.winner(player): return player
               if self.draw(): return None


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
        self.__ply__ = self.__ply__(0, 0, [(dx,dy) for dx in range(0,6) for dy in range(0,6)])

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
def competition(show = False):
    results = {'X': 0, 'O': 0, ' ': 0}
    i = 0
    while i < 100:
        playerX = RandomPlayer('X')
        playerO = StubbornPlayer('O')

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

competition(show=False)



