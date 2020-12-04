import numpy as np
import pgn
import chess
import pickle
import copy
from CNN.src.FastNet.util import *

DATA = "../data/Player Profiles.pgn"
NUM_TRAIN_GAMES = 12000
NUM_TEST_GAMES = 4000

print("Loading PGN file...")

games = get_all_games(DATA)
games = games[NUM_TRAIN_GAMES:NUM_TRAIN_GAMES+NUM_TEST_GAMES]

print("Finished loading the PGN file.")
print("Total number of games: %d" % len(games))

X_white_test = []
X_black_test = []
y_white_test = []
y_black_test = []

for index, game in enumerate(games):
	if index % 100 == 0:
		print("Processed %d games out of %d" % (index, NUM_TEST_GAMES))

	board = chess.Board()
	moves = game.moves

	if 'Alekhine' in game.white:
		w_player = 'Alekhine'
	elif 'Botvinnik' in game.white:
		w_player = 'Botvinnik'
	elif 'Capablanca' in game.white:
		w_player = 'Capablanca'
	elif 'Euwe' in game.white:
		w_player = 'Euwe'
	elif 'Karpov' in game.white:
		w_player = 'Karpov'
	elif 'Kasparov' in game.white:
		w_player = 'Kasparov'
	elif 'Petrosian' in game.white:
		w_player = 'Petrosian'
	elif 'Smyslov' in game.white:
		w_player = 'Smyslov'
	elif 'Spassky' in game.white:
		w_player = 'Spassky'
	elif 'Tal' in game.white:
		w_player = 'Tal'
	else:
		w_player = 'Other'

	if 'Alekhine' in game.black:
		b_player = 'Alekhine'
	elif 'Botvinnik' in game.black:
		b_player = 'Botvinnik'
	elif 'Capablanca' in game.black:
		b_player = 'Capablanca'
	elif 'Euwe' in game.black:
		b_player = 'Euwe'
	elif 'Karpov' in game.black:
		b_player = 'Karpov'
	elif 'Kasparov' in game.black:
		b_player = 'Kasparov'
	elif 'Petrosian' in game.black:
		b_player = 'Petrosian'
	elif 'Smyslov' in game.black:
		b_player = 'Smyslov'
	elif 'Spassky' in game.black:
		b_player = 'Spassky'
	elif 'Tal' in game.black:
		b_player = 'Tal'
	else:
		b_player = 'Other'

	for move_index, move in enumerate(moves):
		if move[0].isalpha():  # check if move is SAN
			if move_index % 2 == 0:
				color = 'white'
				im = convert_bitboard_to_image(board)
			else:
				color = 'black'
				im = flip_image(convert_bitboard_to_image(board))
				im = flip_color(im)

			im = np.rollaxis(im, 2, 0)  # to get into form (C, H, W)

			# Filling the X_test array
			if color == 'white':
				X_white_test.append(im)
			else:
				X_black_test.append(im)
			y_white_test.append(game.white)
			y_black_test.append(game.black)

			board.push_san(move)

X_white_test = np.array(X_white_test)
X_black_test = np.array(X_black_test)
y_white_test = np.array(y_white_test)
y_black_test = np.array(y_black_test)


print("Processed %d games out of %d" % (NUM_TEST_GAMES, NUM_TEST_GAMES))
print("Saving test data...")

output = open('X_white_test_%d.pkl' % NUM_TEST_GAMES, 'wb')
pickle.dump(X_white_test, output)
output.close()

output = open('X_black_test_%d.pkl' % NUM_TEST_GAMES, 'wb')
pickle.dump(X_black_test, output)
output.close()

output = open('y_white_test_%d.pkl' % NUM_TEST_GAMES, 'wb')
pickle.dump(y_white_test, output)
output.close()

output = open('y_black_test_%d.pkl' % NUM_TEST_GAMES, 'wb')
pickle.dump(y_black_test, output)
output.close()

print("Done!")
