import cv2
import numpy as np
import os
import time
from ChessPieceLocate import ChessPieceLocator
from classifier import Chinese_chess_classifier
from digital_board_simulator import Digital_board_simulator
import keyboard
from PIL import Image

SAVE_PATH = 'save_path\\2.jpg'


# function used to generate the digital board model
def place_chess_pieces(class_list, grid_position, base_dir = "gui_src"):
    
    piece_size=(100, 100)
    offset=(50, 50)
    grid_size=100
    board_filename='chessboard_template.png'
    
    # Construct the full path for the chessboard image
    board_file_path = os.path.join(base_dir, board_filename)
    
    # Load the chessboard
    board = Image.open(board_file_path)

    for piece, position in zip(class_list, grid_position):

        # Construct the full path for the piece image
        piece_image_path = os.path.join(base_dir, f"{piece}.png")

        # Load and resize the piece image
        piece_image = Image.open(piece_image_path).resize(piece_size)

        # Calculate the position
        x, y = position
        top_left_x = offset[0] + x * grid_size - piece_size[0] // 2
        top_left_y = offset[1] + y * grid_size - piece_size[1] // 2

        # Paste the piece image onto the board
        board.paste(piece_image, (top_left_x, top_left_y), piece_image)
        board_numpy = np.array(board)
        board_rgb = cv2.cvtColor(board_numpy, cv2.COLOR_BGR2RGB)
        board_final = cv2.resize(board_rgb, (400, 360))

    x, y = position
    top_left_x = offset[0] + x * grid_size - piece_size[0] // 2
    top_left_y = offset[1] + y * grid_size - piece_size[1] // 2

    return board_final


def display_warning(img, msg):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 0.5
    font_color = (255, 255, 255)  
    font_thickness = 1
    x, y = 10, 50  
    cv2.putText(img, msg, (x, y), font, font_scale, font_color, font_thickness)
    return img


def display_msg(img, msg):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 0.8
    font_color = (255, 0, 0)  
    font_thickness = 1
    x, y = 10, 50  
    cv2.putText(img, msg, (x, y), font, font_scale, font_color, font_thickness)
    return img




def main():

    # initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    success, img  = cap.read()
    cv2.imwrite(SAVE_PATH, img)
    board_model = np.zeros((360, 400))  # initialize board model

    # create the objects
    processor = ChessPieceLocator(SAVE_PATH)
    my_classifer = Chinese_chess_classifier()
    my_digital_board_simulator = Digital_board_simulator()
    time.sleep(5)

    print ('press "w" to begin')
    i = 1

    # main loop
    while True:
        keyboard.wait('w')
        print(f'round: {i}')
        i += 1
        success, img  = cap.read()
        cv2.imwrite(SAVE_PATH, img)
        
        processor.update_img(SAVE_PATH)
        cropped_images, grid_positions, status = processor.pos_normalize()

        # the case when these is no chess move
        if status == 2:
            warning = 'No movement detected.'
            print(warning)
            board_model = display_warning(np.zeros((360, 400)), warning)

        # the case when these is something blocking the board
        elif status == 3:
            warning = 'Obstruction detected, invalid detection!'
            print(warning)
            board_model = display_warning(np.zeros((360, 400)), warning)

        # normal case, then do the chess piece classification
        elif status == 1:      
            # classification
            class_list = my_classifer.overall_classification(cropped_images)

            # update simulator state and refresh digital board 
            my_digital_board_simulator.update_state(grid_positions, class_list)
            my_digital_board_simulator.refresh_digital_board()
            my_digital_board_simulator.display_board()
            msg = my_digital_board_simulator.check_legitimate()
            board_model = place_chess_pieces(class_list, grid_positions, base_dir = "gui_src")
            if msg != 'no error':
                board_model = display_msg(board_model, msg)

        # display the digital board model
        resized_img_new = img[:, 500:1700, :]
        resized_img_new = cv2.resize(resized_img_new, (400, 360))
        cv2.imshow('webcam capture', resized_img_new)
        cv2.imshow('board_model', board_model)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Press "W" for the next detection')
        keyboard.wait('w')




if __name__ == "__main__":
    main()
    

