import cv2
import cv2.aruco as aruco

# Create ChArUco board, which is a set of Aruco markers in a chessboard setting
# meant for calibration
# the following call gets a ChArUco board of tiles 6 wide X 8 tall
gridboard = aruco.CharucoBoard_create(
        squaresX=6, 
        squaresY=8, 
        squareLength=0.04, 
        markerLength=0.02, 
        dictionary=aruco.Dictionary_get(aruco.DICT_6X6_250))

# Create an image from the gridboard
img = gridboard.draw(outSize=(988, 1400))
cv2.imwrite("res/charuco_out/test_charuco.jpg", img)

# Display the image
cv2.imshow('Gridboard', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()