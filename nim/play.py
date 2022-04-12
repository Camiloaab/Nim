from nim import train, play


# """"
#     Initial: the human player can choose the initial configuration
#     """
# initial=list(input("Choose the initial configuration as coma separated integers"))    


""""
    Type of play: The human player can choose whether the game is normal play
    (last to move wins) of misere (last to move loses).
    """
game_type=2*int(input("Type 0 for normal play or 1 for Misere "))-1
ai = train(10000,game_type)
play(ai,game_type)