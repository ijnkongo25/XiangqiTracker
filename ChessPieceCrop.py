import cv2
import numpy as np
import os

class ChessPieceDetector:
    
    def __init__(self, img):
        self.img = img
        self.cropped_images = []
        self.center_positions = []
                
    def detect_chess_pieces(self):
        
        # Use the provided image directly instead of reading from file
        img = self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                   param1=90, param2=30, minRadius=30, maxRadius=60)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for i, (x, y, r) in enumerate(circles):
                roi = img[y-r:y+r, x-r:x+r]
                mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                
                cv2.circle(mask, (r, r), r, (255, 255, 255), -1)
                masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
                
                final_roi = cv2.resize(masked_roi, (100, 100))
                
                self.cropped_images.append(final_roi)
                self.center_positions.append((x, y))
        
        return self.cropped_images, self.center_positions
