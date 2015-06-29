import random

# ---------------------------------------------------------------------
class Game:
    def __init__(self):
        self.field = [[0] * 6 for x in xrange(6)]

    def show(self):
        print '   ' + ' '.join([' '+str(x)+' ' for x in xrange(0,6)])
        print '  -------------------------'
        for idx, line in enumerate(self.field):
            print idx,  '|' + \
                '|'.join([ ' X ' if cell==1 else ' O ' \
                                 if cell==2 else '   ' for cell in line]) + \
                '|'
            print '  -------------------------'

    def set(self, xy, val):
        self.field[xy[0]][xy[1]] = 1 if val == 'X' else 2

    def get(self, xy):
        val = self.field[xy[0]][xy[1]]
        return 'X' if val == 1 else 'O' if val == 2 else ' '

    def available(self):
        return [(y, x) for y, row in enumerate(self.field) for x, v in enumerate(row) if v == 0]

    def winner(self, player):
        n = 4
        str = player.role * n
        for x in range(0, 6 - n + 1):
            for y in range(0, 6 - n + 1):
                hor = [(x, y + i) for i in range(0, n)]
                ver = [(x + i, y) for i in range(0, n)]
                dia = [(x + i, y + i) for i in range(0, n)]

                hvd = [ ''.join([ self.get(xy) for xy in four ]) for four in [hor, ver, dia] ]

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
def competition():
    results = {'X': 0, 'O': 0, ' ': 0}
    i = 0
    while i < 100:
        i+=1
        tictactoe = Game()
        winner = tictactoe.play([ RandomPlayer('X'), RandomPlayer('O') ], False)
        if winner is None:
            results[' '] = results[' '] + 1
        else:
            results[winner.role] = results[winner.role] + 1

    print 'X: ', results['X']
    print 'O: ', results['O']
    print ' : ', results[' ']




# ---------------------------------------------------------------------
#tictactoe.play([ HumanPlayer('X'), RandomPlayer('O') ] )

results = {'X': 0, 'O': 0, ' ': 0}
i = 0
while i < 100:
   i+=1
   tictactoe = Game()
   winner = tictactoe.play([ RandomPlayer('X'), RandomPlayer('O') ], False)
   if winner is None:
       results[' '] = results[' '] + 1
   else:
       results[winner.role] = results[winner.role] + 1

print 'X: ', results['X']
print 'O: ', results['O']
print ' : ', results[' ']

