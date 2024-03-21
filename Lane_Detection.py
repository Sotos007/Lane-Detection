import cv2
import numpy as np
import os.path


# Function to check if the video file exists
def video_exists(video_path):
    return os.path.isfile(video_path)


# Function to get a valid video path from user input
def get_valid_video_path():
    video_path = input("ENTER THE PATH OF THE VIDEO FOR LANE DETECTION: ")
    while not video_exists(video_path):
        print("The video file does not exist. Please enter a valid video path.")
        video_path = input("ENTER THE PATH OF THE VIDEO FOR LANE DETECTION: ")
    return video_path


# Function to define a region of interest in the image
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Function to draw lines on the image
def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 230, 0), thickness=3)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


# Function to process each frame of the video
def process(image):
    height, width, _ = image.shape
    region_of_interest_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines


# Check the existence of the video file before opening the video capture
video_path = get_valid_video_path()
cap = cv2.VideoCapture(video_path)

paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process(frame)
        cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    # Key for quitting
    if key == ord('q'):
        break
    # Key for pausing
    elif key == ord('s'):
        paused = True
    # Key for resuming
    elif key == ord('r'):
        paused = False

cap.release()
cv2.destroyAllWindows()
