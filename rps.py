from __future__ import division, print_function
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO

import numpy as np
import random
import json


# Takes a dictionary of players as keys and weapons as values
# Returns the key of the winning player
def winner(game):
    winning_definitions = {frozenset(['paper', 'rock']): 'paper',
                           frozenset(['rock', 'scissors']): 'rock',
                           frozenset(['scissors', 'paper']): 'scissors',}
    winning_weapon = winning_definitions.get(frozenset([game['Human'], game['Computer']]), 'draw')
    for player, weapon in game.items():
        if winning_weapon == 'draw':
            return 'draw'
        elif winning_weapon == weapon:
            return player.lower()

# Returns the choice entered at the command line
def human_choice():
    valid_options = frozenset(['rock', 'paper', 'scissors'])
    while True:
        user_input = raw_input('Please enter your choice: rock, paper, scissors: ').lower().strip()
        if user_input in valid_options:
            return user_input
        else:
            print('Entry not valid.')

# Returns a random weapon
def random_choice():
    print('playing random')
    return random.choice(['rock', 'paper', 'scissors'])

# Returns weapon based on http://www.businessinsider.com/how-to-beat-anyone-at-rock-paper-scissors-2014-5
def wang_choice(last_game):
    human_won = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    human_lost = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    if last_game['Champion'] == 'human':
        return human_won[last_game['Human']]
    else:
        return human_lost.get(last_game['Human'], random_choice())

# Returns choice based on decision tree
def tree_choice(last_game):
    offensive_definitions = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    for value in last_game.values():
        if value == None:
            null_found = True
            break
        else:
            null_found = False
    if null_found:
        print('found null - playing wang')
        return wang_choice(last_game)
    else:
        print('playing tree')
        weapons_encoder = preprocessing.LabelEncoder()
        weapons_encoder.fit(['paper', 'rock', 'scissors'])
        champion_encoder = preprocessing.LabelEncoder()
        champion_encoder.fit(['computer', 'human', 'draw'])
        samples = []
        labels = []
        #get data
        with open('data', 'r') as f:
            for line in f.readlines():
                # preprocess data
                game = json.loads(line)
                del game['Game']
                del game['Computer']
                del game['Champion']
                human_older = weapons_encoder.transform([game['Human_Older']])
                computer_older = weapons_encoder.transform([game['Computer_Older']])
                champion_older = champion_encoder.transform([game['Champion_Older']])
                sample = human_older + computer_older + champion_older
                samples.append(sample)
                labels.append(weapons_encoder.transform([game['Human']]))
        # train the model with the data
        tree = DecisionTreeClassifier()
        tree = tree.fit(samples, labels)

        # print tree
        with open('tree.dot', 'w') as f:
            f = export_graphviz(tree, out_file=f)

        # predict and return the next choice
        human = weapons_encoder.transform(last_game['Human'])
        computer = weapons_encoder.transform(last_game['Computer'])
        champion = champion_encoder.transform(last_game['Champion'])
        test_data = [human + computer + champion]
        test_data = np.array(test_data).reshape(1, -1)
        prediction = tree.predict(test_data, labels)
        prediction = weapons_encoder.inverse_transform(prediction)
        return offensive_definitions[prediction[0]]

# Returns game selection
def game_type():
    valid_options = frozenset(['random', 'wang', 'tree'])
    while True:
        user_input = raw_input('Please enter your choice of game: Random, Wang, or Tree: ').lower().strip()
        if user_input in valid_options:
            return user_input
        else:
            print('Entry not valid.')

# Create and store a new data record
def process_results(game, last_game):
    # create a new last_game
    last_game['Game_Older'] = last_game['Game']
    last_game['Game'] = game['Game']
    last_game['Computer_Older'] = last_game['Computer']
    last_game['Computer'] = game['Computer']
    last_game['Human_Older'] = last_game['Human']
    last_game['Human'] = game['Human']
    last_game['Champion_Older'] = last_game['Champion']
    last_game['Champion'] = game['Champion']
    # store the new last_game
    for value in last_game.values():
        if value == None:
            save = False
            break
        else:
            save = True
    if save:
        with open('data', 'a') as f:
            f.write(json.dumps(last_game) + '\n')
    #return the new last_game
    return last_game

# Runs the main program
def main():
    # Initialize the game
    total_games = 0
    human_wins = 0
    last_game = {'Game': None, 'Computer': None, 'Human': None, 'Champion': None,
                 'Game_Older': None, 'Computer_Older': None, 'Human_Older': None, 'Champion_Older': None}
    # Play the game
    g_t = game_type()
    more = True
    while more:
        print('\nOld last_game: ', end='')
        print(last_game)
        if g_t == 'random':
            game = {'Game': 'random'}
            game['Computer'] = random_choice()
        elif g_t == 'wang':
            game = {'Game': 'wang'}
            game['Computer'] = wang_choice(last_game)
        elif g_t == 'tree':
            game = {'Game': 'tree'}
            game['Computer'] = tree_choice(last_game)
        else:
            print('Game Type Error - Tell Daniel')
            break
        game['Human'] = human_choice()
        game['Champion'] = winner(game)
        print('Current game: ', end='')
        print(game)

        # Process the data and create a new last game
        last_game = process_results(game, last_game)
        print('New last_game: ', end='')
        print(last_game)

        # Report on the game
        if game['Champion'] != 'draw':
            total_games += 1
            if game['Champion'] == 'human':
                human_wins += 1
            print('\n' + game['Champion'] + ' wins!')
            print('Human Win Rate: {0:.0f}%'.format(human_wins/total_games * 100))
            print('Total Game: ', end='')
            print(total_games)
        else:
            print('\nDraw')

        # Ask if humans want to go again
        if raw_input('\nContinue? n = no, anything else = yes: ') == 'n':
            more = False

if __name__ == '__main__':
    main()