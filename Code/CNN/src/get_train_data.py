import numpy as np
import pgn
import chess
import pickle
import copy
from CNN.src.FastNet.util import *

DATA = "../data/Player Profiles.pgn"
NUM_GAMES = 12000

print("Loading PGN file...")

games = get_all_games(DATA)
games = games[:NUM_GAMES]

print("Finished loading the PGN file.")
print("Total number of games: %d" % len(games))

# Preparing general training arrays
X_train, y_train = [], []
# Preparing champion training arrays
p1_X, p2_X, p3_X, p4_X, p5_X = [], [], [], [], []
p6_X, p7_X, p8_X, p9_X, p10_X = [], [], [], [], []
p1_y, p2_y, p3_y, p4_y, p5_y = [], [], [], [], []
p6_y, p7_y, p8_y, p9_y, p10_y = [], [], [], [], []
for index, game in enumerate(games):
	if index % 100 == 0:
		print("Processed %d games out of %d" % (index, NUM_GAMES))

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

	if b_player == 'Other':
		if w_player == 'Alekhine':
			champidx = 1
		elif w_player == 'Botvinnik':
			champidx = 2
		elif w_player == 'Capablanca':
			champidx = 3
		elif w_player == 'Euwe':
			champidx = 4
		elif w_player == 'Karpov':
			champidx = 5
		elif w_player == 'Kasparov':
			champidx = 6
		elif w_player == 'Petrosian':
			champidx = 7
		elif w_player == 'Smyslov':
			champidx = 8
		elif w_player == 'Spassky':
			champidx = 9
		elif w_player == 'Tal':
			champidx = 10
	else:
		if b_player == 'Alekhine':
			champidx = 1
		elif b_player == 'Botvinnik':
			champidx = 2
		elif b_player == 'Capablanca':
			champidx = 3
		elif b_player == 'Euwe':
			champidx = 4
		elif b_player == 'Karpov':
			champidx = 5
		elif b_player == 'Kasparov':
			champidx = 6
		elif b_player == 'Petrosian':
			champidx = 7
		elif b_player == 'Smyslov':
			champidx = 8
		elif b_player == 'Spassky':
			champidx = 9
		elif b_player == 'Tal':
			champidx = 10
		else:
			champidx = 0

	for move_index, move in enumerate(moves):
		if move[0].isalpha(): # check if move is SAN		
				
			from_to_chess_coords = board.parse_san(move)
			from_to_chess_coords = str(from_to_chess_coords)

			from_chess_coords = from_to_chess_coords[:2]
			to_chess_coords = from_to_chess_coords[2:4]
			from_coords = chess_coord_to_coord2d(from_chess_coords)
			to_coords = chess_coord_to_coord2d(to_chess_coords)
						
			if move_index % 2 == 0:
				color = 'white'
				im = convert_bitboard_to_image(board)
			else:
				color = 'black'
				im = flip_image(convert_bitboard_to_image(board))
				im = flip_color(im)
				from_coords = flip_coord2d(from_coords)
				to_coords = flip_coord2d(to_coords)

			from_coords = flatten_coord2d(from_coords)
			to_coords = flatten_coord2d(to_coords)

			im = np.rollaxis(im, 2, 0) # to get into form (C, H, W)

			board.push_san(move)

			# Filling the X_train and y_train array
			if color == 'white':
				player = w_player
			else:
				player = b_player
			X_train.append(im)
			y_train.append(player)


			# Filling the p_X and p_y array
			if champidx != 0:
				p_X = "p%d_X" % (champidx)
				p_X = eval(p_X)
				p_X.append(im)
				p_y = "p%d_y" % (champidx)
				p_y = eval(p_y)
				p_y.append(player)
			else:
				continue


# Move-out
X_train, y_train = np.array(X_train), np.array(y_train)

# Move-in
p1_X, p2_X, p3_X = np.array(p1_X), np.array(p2_X), np.array(p3_X)
p4_X, p5_X, p6_X = np.array(p4_X), np.array(p5_X), np.array(p6_X)
p7_X, p8_X, p9_X, p10_X = np.array(p7_X), np.array(p8_X), np.array(p9_X), np.array(p10_X)

p1_y, p2_y, p3_y = np.array(p1_y), np.array(p2_y), np.array(p3_y)
p4_y, p5_y, p6_y = np.array(p4_y), np.array(p5_y), np.array(p6_y)
p7_y, p8_y, p9_y, p10_y = np.array(p7_y), np.array(p8_y), np.array(p9_y), np.array(p10_y)

print("Processed %d games out of %d" % (NUM_GAMES, NUM_GAMES))
print("Saving data...")

print("Saving X_train array...")
output = open('X_train_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(X_train, output)
output.close()

print("Saving y_train array...")
output = open('y_train_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(y_train, output)
output.close()

for i in range(10):
	output_array = "p%d_X" % (i+1)
	print("Saving %s array..." % output_array)
	output_array = eval(output_array)
	output = open('p%d_X_%d.pkl' % (i+1, NUM_GAMES), 'wb')
	pickle.dump(output_array, output)
	output.close()

	output_array = "p%d_y" % (i+1)
	print("Saving %s array..." % output_array)
	output_array = eval(output_array)
	output = open('p%d_y_%d.pkl' % (i+1, NUM_GAMES), 'wb')
	pickle.dump(output_array, output)
	output.close()

print("Done!")
