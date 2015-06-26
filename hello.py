

class Game:
    def __init__(self, name):
        self.field = [[0] * 6 for x in xrange(6)]

    def show(self):
        print '   ' + ' '.join([' '+str(x)+' ' for x in xrange(1,7)])
        print '  -------------------------'
        for idx, line in enumerate(self.field):
            print idx + 1,  '|' + \
                '|'.join([ ' X ' if cell==1 else ' O ' \
                                 if cell==2 else '   ' for cell in line]) + \
                '|'
            print '  -------------------------'

    def set(self, cell, val):
        self.field[cell[0] - 1][cell[1] - 1] = 1 if val == 'X' else 2



'''
var = 'something'
while var != '':
    var = raw_input("Please enter something: ")
    print "you entered", var
'''

game = Game('tictactoe')

game.set((1,2), 'X')
game.set((3,4), 'O')

game.show()

