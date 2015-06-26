

class Game:
	def __init__(self, name):
		self.field = [[0] * 6 for x in xrange(6)]

	def show(self):
		print '----------------------'
		for line in self.field:
		    print '|', ''.join([ ' X ' if cell==1 else ' 0 ' if cell==2 else '   ' for cell in line]), \
			      '|'
		print '----------------------'


#		    for cell in line:
#		        print ' X ' if cell == 1 else ' O ' if cell == 2 else '   '

'''
var = 'something'
while var != '':
	var = raw_input("Please enter something: ")
	print "you entered", var
'''

game = Game('tictactoe')


game.field[0][1] = 1
game.field[2][3] = 2


game.show()

