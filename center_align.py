import math

class CableCarBot:
    def move(self, direction):
        # Example method to move the cable car
        print(f"Moving {direction}")

def centre_align(p1, p2, p3, p4, image_width, image_height, bot):
    # Calculate the center of the bounding box
    center_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
    center_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
    
    # Image center
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    
    # Calculate distance from the bounding box center to the image center
    dist_x = center_x - image_center_x
    dist_y = center_y - image_center_y
    
    # Define a threshold to determine movement
    threshold = 10  # Adjust as needed
    
    # Determine movement direction
    if abs(dist_y) > threshold:
        if dist_y < 0:  # Sunflower is above the center
            bot.move('forward')
        elif dist_y > 0:  # Sunflower is below the center
            bot.move('backward')
    else:
        print("Sunflower is centered. No movement required.")

'''
# Example usage
bot = CableCarBot()

# Example bounding box points and image dimensions
p1 = (100, 50)  # Top-left corner
p2 = (200, 50)  # Top-right corner
p3 = (200, 150) # Bottom-right corner
p4 = (100, 150) # Bottom-left corner

image_width = 640
image_height = 480

centre_align(p1, p2, p3, p4, image_width, image_height, bot)
'''
