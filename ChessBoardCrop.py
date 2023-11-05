from pickle import NONE
import cv2 as cv
import numpy as np

class ChessBoardProcessor:
    def __init__(self, image_path):
        self.image_path = image_path

    def update(self, image_path):
        self.image_path = image_path

    #  Show the picture in 700*700
    def show_pic(self, image):
        self.ChessBoardProcessor_show_pic(image)

    # Pre-process: Adaptive Histogram Equalization
    def preprocess_image(self):
        return self.ChessBoardProcessor_preprocess_image()

    # select red region
    def select_red_region(self, image):
        return self.ChessBoardProcessor_select_red_region(image)

    # Find the corners
    def find_red_corners(self, red_region, offset):
        return self.ChessBoardProcessor_find_red_corners(red_region, offset)
    
    def mark_and_show_corners(self, image, corners):
        return self.ChessBoardProcessor_mark_and_show_corners(image, corners)

    def crop_rectangle(self, image, corners):
        return self.ChessBoardProcessor_crop_rectangle(image, corners)

    def adjust_brightness(self, image, brightness_increase=30):
        return self.ChessBoardProcessor_adjust_brightness(image, brightness_increase)
    
    # It returns all the locations of drop point
    def locate_intersection_points(self, image,corners):
         return self.ChessBoardProcessor_locate_intersection_points(image,corners)


    # Main function 
    # It will output an unit8 picture and an array of 4 corners
    counter = 0 # edit later, for naming part
    def process_image(self):
        return self.ChessBoardProcessor_process_image()


    ## Functions
    def ChessBoardProcessor_show_pic(self, image):
            cv.namedWindow('image', cv.WINDOW_NORMAL)
            cv.resizeWindow('image', 700, 700)
            cv.imshow('image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()


    def ChessBoardProcessor_preprocess_image(self):
            image = cv.imread(self.image_path, cv.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to load image.")
            
            # Convert image to YCrCb color space
            ycrcb_image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
            
            # Split the channels
            y, cr, cb = cv.split(ycrcb_image)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the Y channel
            clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
            clahe_y = clahe.apply(y)
            
            # Merge the channels back
            clahe_ycrcb_image = cv.merge([clahe_y, cr, cb])
            
            # Convert back to BGR color space
            final_image = cv.cvtColor(clahe_ycrcb_image, cv.COLOR_YCrCb2BGR)
            return final_image


    def ChessBoardProcessor_select_red_region(self,image):
        # Load image
        if image is None:
            raise ValueError("Failed to load image.")
        
        # Convert to HSV color space
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Define the range of red color
        lower_red_1 = np.array([0, 100, 100])       # Decreased saturation and brightness
        upper_red_1 = np.array([15, 255, 255])      # Increased hue
        lower_red_2 = np.array([160, 100, 100])     # Decreased saturation, brightness, and hue
        upper_red_2 = np.array([180, 255, 255])

        # Create a mask for the red color
        mask1 = cv.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv.inRange(hsv, lower_red_2, upper_red_2)
        mask = mask1 + mask2

        # Bitwise-AND mask and original image
        red_region = cv.bitwise_and(image, image, mask=mask)

        return red_region


    def ChessBoardProcessor_find_red_corners(self,red_region,offset):
        # Convert the red region to grayscale
        gray_red_region = cv.cvtColor(red_region, cv.COLOR_BGR2GRAY)
        
        # Binarize the grayscale red region
        _, binarized_red_region = cv.threshold(gray_red_region, 1, 255, cv.THRESH_BINARY)

        kernel = np.ones((15, 15), np.uint8)  # You can change the kernel size as needed
        dilated_img = cv.dilate(binarized_red_region, kernel, iterations=3)

        if_clear = self.detect_obstruction(dilated_img)

        # Find the contours in the binarized red region
        contours, _ = cv.findContours(dilated_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        contour = max(contours, key=cv.contourArea)
        
        # Get the coordinates of the contour points
        coords = np.array(contour).squeeze()
        
        # Find the corners
        top_left = coords[np.argmin(coords[:, 0] + coords[:, 1])] - [offset, offset]
        bottom_left = coords[np.argmin(coords[:, 0] - coords[:, 1])] + [-offset, offset]
        bottom_right = coords[np.argmax(coords[:, 0] + coords[:, 1])] + [offset, offset]
        top_right = coords[np.argmax(coords[:, 0] - coords[:, 1])] + [offset, -offset]

        corners = [top_left, top_right, bottom_right, bottom_left]
        
        return corners, if_clear


    def ChessBoardProcessor_crop_rectangle(self,image, corners):
        # Sort corners
        corners = np.array(corners)
        rect = np.zeros((4, 2), dtype="float32")

        # Top-left corner will have the smallest sum, 
        # Bottom-right corner will have the largest sum
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]

        # Top-right will have the smallest difference
        # Bottom-left will have the largest difference
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]

        # Determine the width and height of the new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # Set up the destination points for the perspective transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ], dtype="float32")

        # Compute the perspective transform matrix and then apply it
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped


    def ChessBoardProcessor_adjust_brightness(self,image, brightness_increase=30):
        # Define the HSV color bounds for the wooden chess pieces and board
        lower_bound = np.array([10, 50, 20])  # example values
        upper_bound = np.array([50, 255, 255])  # example values

        # Convert to HSV color space
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
        # Define a mask based on the color bounds
        mask = cv.inRange(hsv, lower_bound, upper_bound)
        
        # Increase the brightness of the masked regions
        hsv[:, :, 2] = cv.addWeighted(hsv[:, :, 2], 1, (mask/255).astype(np.uint8), brightness_increase, 0)

        # Convert back to BGR color space
        final_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        return final_image

    # Mark the corners
    def ChessBoardProcessor_mark_and_show_corners(self, image, corners):
        # Mark each corner with a circle
        for (x, y) in corners:
            cv.circle(image, (int(round(x)), int(round(y))), 10, (0, 0, 255), -1)



    def ChessBoardProcessor_locate_intersection_points(self, image, corners):
        # Calculate the offset from red corners to board corners
        top_left = corners[0]
        top_right = corners[1]
        bottom_right = corners[2]
        bottom_left = corners[3]

        offset_width = int((top_right[0] - top_left[0]) / 10 /2)
        offset_height = int((bottom_left[1] - top_left[1]) / 9 /2)

        # Aplly the offset
        top_left[0] += offset_width
        top_left[1] += offset_height
        top_right[0] -= offset_width
        top_right[1] += offset_height
        bottom_left[0] += offset_width
        bottom_left[1] -= offset_height
        bottom_right[0] -= offset_width
        bottom_right[1] -= offset_height

        corners = [top_left, top_right, bottom_right, bottom_left]
        # self.mark_and_show_corners(image,corners)

        # Calculate grid width of the board
        grid_width = int((top_right[0] - top_left[0]) / 9)
        grid_height = int((bottom_left[1] - top_left[1]) / 8)
        
        
        points = []

        for i in range(0, 9):  # vertical lines
            for j in range(0, 10):  # horizontal lines
                x = int(j * grid_width) + top_left[0]
                y = int(i * grid_height) + top_left[1]
                points.append((x, y))

        return points


    # Main process function
    def ChessBoardProcessor_process_image(self):
            '''
            return status
                1: normal case
                2: no moved chess
                3: imcomplete board
            '''
            # Calibration
            preprocessed_image = self.preprocess_image()
            red_area = self.select_red_region(preprocessed_image)
            # self.show_pic(red_area)

            corners, if_clear = self.find_red_corners(red_area, 3)    
            
            cropped_img = self.crop_rectangle(preprocessed_image, corners)
            
            # Brighten the cropped image
            brightened_cropped_image = self.adjust_brightness(cropped_img)

            # Clip the pixel values to be within 0-255
            # Change the data type to uint8 
            brightened_cropped_image = np.clip(brightened_cropped_image, 0, 255)
            brightened_cropped_image = brightened_cropped_image.astype(np.uint8)

            status = self.check_board_status(brightened_cropped_image, if_clear)
            if status == 1:

                # Cropped corners
                cropped_red_area = self.select_red_region(brightened_cropped_image)
                cropped_corners, if_clear = self.find_red_corners(cropped_red_area, -15)
                self.mark_and_show_corners(cropped_img,cropped_corners)

                # Save cropped and brightened image
                filename = f'Save\cropped_chessboard_{ChessBoardProcessor.counter}.jpg'
                cv.imwrite(filename, brightened_cropped_image)
                ChessBoardProcessor.counter += 1

                # Calculate all the intersection points' locations
                intersection_points = self.locate_intersection_points(brightened_cropped_image,cropped_corners)

            elif status == 2:
                intersection_points = None
                brightened_cropped_image = None
                print ('chess board unchanged')
            elif status == 3:
                intersection_points = None
                brightened_cropped_image = None
                print ('WARNING: chess board is incomplete')

            return intersection_points,brightened_cropped_image, status
    

    def check_board_status(self, board_img_current, if_clear):
        status = 1
        if not if_clear:
            status = 3
        return status
             
             
    def detect_obstruction(self, image): 
        # Finding contours
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area size, and select the largest contour, which should be the outer boundary
        contour = sorted(contours, key=cv.contourArea, reverse=True)[0]

        # Get the dimensions of the image
        h, w = image.shape[:2]

        # Create a completely black background
        background = np.zeros((h, w), dtype=np.uint8)

        # Fill the background with the found contour
        cv.drawContours(background, [contour], 0, (255), thickness=cv.FILLED)

        # Compare the original image with the filled background
        area_backgound = np.count_nonzero(background)
        area_binarized = np.count_nonzero(image)
        difference = area_backgound - area_binarized

        # If there is a difference, it is considered to be obstructed
        if difference < 0:
            pass
            return False
        else:
            pass
            return True

