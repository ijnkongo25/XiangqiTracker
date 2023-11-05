# import os

import random
import numpy as np


class Digital_board_simulator:
    def __init__(self):
        self.empty_board = [['   ' for _ in range(9)] for _ in range(10)]
        self.digital_board = [['   ' for _ in range(9)] for _ in range(10)]
        self.previous_position_list = []
        self.previous_class_list = []
        self.current_position_list = []
        self.current_class_list = []


    # update the class variables based
    def update_state(self, position_list, class_list):
        """
        param examples:
        position_list = [[0,0],[8,4],[2,1]]
        class_list    = ["cannon_black", "guard_black", "knight_black"]
        """
        self.previous_position_list = self.current_position_list
        self.previous_class_list = self.current_class_list
        self.current_position_list = position_list
        self.current_class_list = class_list


    # method: fresh the digital board based on the new chess positions
    def refresh_digital_board(self):

        if len(self.current_position_list) != len(self.current_class_list):
            print('list lengths are not equal')
            exit()

        self.digital_board =  [['   ' for _ in range(9)] for _ in range(10)]

        for idx in range(len(self.current_position_list)):
            this_piece_position = self.current_position_list[idx]
            x = this_piece_position[0]
            x = 9-x
            y = this_piece_position[1]
            name = self.simplify_class(self.current_class_list[idx])
            self.digital_board[x][y] = name


    # method: check the legality of the chess move
    def check_legitimate(self):

        position_list = self.current_position_list
        class_list    = self.current_class_list

        # check if the is two lists are in the same lengths
        if len(position_list) != len(class_list):
            print('list lengths are not equal')
            exit()
        
        # if black king is killed, then red team win
        if not ('king_black' in class_list):
            print("red team wins!")
            msg = 'red team wins!'
            return msg

        # if red king is killed, then black team win
        if not ('king_red' in class_list):
            print("black team wins!")
            msg = 'black team wins!'
            return msg


        for idx in range(len(position_list)):
            this_piece_position = position_list[idx]
            x = this_piece_position[0]
            x = 9-x     # flip the x position
            y = this_piece_position[1]
            name = class_list[idx]
            msg = 'no error'
            
            # check if red king is legal
            if name == 'king_red':
                if (y<3) or (y>5) or (x<7):
                    print('WARNING: illegal red king')
                    msg = 'WARNING: illegal red king'
                    return msg
            
            # check if black king is legal
            if name == 'king_black':
                if (y<3) or (y>5) or (x>2):
                    print('WARNING: illegal black king')
                    msg = 'WARNING: illegal black king'
                    return msg

            # check if black guard is legal
            if name == 'guard_black':
                if (y<3) or (y>5) or (x>2):
                    print('WARNING: illegal black guard')
                    msg = 'WARNING: illegal black guard'
                    return msg
            
            # check if red guard is legal
            if name == 'guard_red':
                if (y<3) or (y>5) or (x<7):
                    print('WARNING: illegal red guard')
                    msg = 'WARNING: illegal red guard'
                    return msg

            # check if black elephant is legal
            if name == 'elephant_black':
                if (x>4):
                    print('WARNING: illegal black elephant')
                    msg = 'WARNING: illegal black elephant'
                    return msg
            
            # check if red elephant is legal
            if name == 'elephant_red':
                if (x<5):
                    print('WARNING: illegal red elephant')
                    msg = 'WARNING: illegal red elephant'
                    return msg
        return 'no error'

    
    # method used to display the digital board in cmb window
    def display_board(self):
        for row in self.digital_board:
            print(row)


    # convert the name from label in English to Chinese letter
    def simplify_class(self, class_name):
            if class_name == "cannon_black":
                return "炮b"
            elif class_name == "cannon_red":
                return "炮r"
            elif class_name == "elephant_black":
                return "象b"
            elif class_name == "elephant_red":
                return "相r"
            elif class_name == "guard_black":
                return "士b"
            elif class_name == "guard_red":
                return "士r"
            elif class_name == "king_black":
                return "将b"
            elif class_name == "king_red":
                return "帥r"
            elif class_name == "knight_black":
                return "馬b"
            elif class_name == "knight_red":
                return "馬r"
            elif class_name == "pawn_black":
                return "卒b"
            elif class_name == "pawn_red":
                return "兵r"
            elif class_name == "rook_black":
                return "車b"
            elif class_name == "rook_red":
                return "車r"
            else:
                return "0"




if __name__ == '__main__':

    try_board = Digital_board_simulator()
    position_list = [[0,0],[8,4],[2,1]]
    class_list    = ["cannon_black", "guard_black", "knight_black"]
    
    try_board.update_digital_board(position_list,class_list)
    try_board.display_board()





