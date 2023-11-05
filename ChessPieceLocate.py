import cv2
import numpy as np
import os
from ChessBoardCrop import ChessBoardProcessor
from ChessPieceCrop import ChessPieceDetector
from classifier import Chinese_chess_classifier
import keyboard

class ChessPieceLocator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.chessboard_processor = ChessBoardProcessor(self.image_path)

    def update_img(self, image_path):
        self.image_path = image_path

    def pos_normalize(self):
        self.chessboard_processor.update(self.image_path)
        intersection_points, processed_image, status = self.chessboard_processor.process_image()
        if status == 1:
            grid_width = (intersection_points[1][0] - intersection_points[0][0])
            grid_height = (intersection_points[10][1] - intersection_points[0][1])

            xiangqi_detector = ChessPieceDetector(processed_image)
            cropped_images, center_positions = xiangqi_detector.detect_chess_pieces()

            grid_positions = []
            for center_position in center_positions:
                x_pixel, y_pixel = center_position
                x_grid = round((x_pixel - intersection_points[0][0]) / grid_width)
                y_grid = round((y_pixel - intersection_points[0][1]) / grid_height)
                grid_positions.append([x_grid, y_grid])
        else:
            cropped_images = None
            grid_positions = None
        return cropped_images, grid_positions, status

    def save_cropped_images(self, cropped_images, grid_positions):
        self.clear_directory()
        for idx, (cropped_image, grid_position) in enumerate(zip(cropped_images, grid_positions)):
            filename = f'CroppedPieces/Piece{idx + 1}_PosX{grid_position[0]}_PosY{grid_position[1]}.png'
            cv2.imwrite(filename, cropped_image)

    def clear_directory(self):
        if not os.path.exists('CroppedPieces'):
            os.mkdir('CroppedPieces')
        else:
            for filename in os.listdir('CroppedPieces'):
                os.remove(os.path.join('CroppedPieces', filename))
                
                
    def overlay_chessboard(self, cropped_images, grid_positions):

        chessboard_template_path = 'chessboard_template.png'
        chessboard_image = cv2.imread(chessboard_template_path)
        if chessboard_image is None:
            raise ValueError(f"Failed to load chessboard template at {chessboard_template_path}")

        for cropped_image, grid_position in zip(cropped_images, grid_positions):
            x_grid, y_grid = grid_position
            grid_width, grid_height = 100, 100 

            x_pixel = 50 + x_grid * grid_width - cropped_image.shape[1] // 2
            y_pixel = 50 + y_grid * grid_height - cropped_image.shape[0] // 2

            if cropped_image.shape[:2] != (grid_height, grid_width):
                cropped_image = cv2.resize(cropped_image, (grid_width, grid_height))

            chessboard_image[y_pixel:y_pixel + grid_height, x_pixel:x_pixel + grid_width] = cropped_image

        result_filename = 'generated_chessboard.png'
        cv2.imwrite(result_filename, chessboard_image)


if __name__ == "__main__":
    image_path = "E:\\AMME4710_major\\CNN\\save_path\\2.jpg"
    processor = ChessPieceLocator(image_path)
    cropped_images, grid_positions = processor.pos_normalize()
    processor.save_cropped_images(cropped_images, grid_positions)
    this_img = cropped_images[0]

    
    for idx, grid_position in enumerate(grid_positions):
        pass
    processor.overlay_chessboard(cropped_images, grid_positions)


