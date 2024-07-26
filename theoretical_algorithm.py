import json
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

import math

import time
import math

config = {
   "phone-to-call": "+91 9972910248"
}

matching_config = {
    "main_image_path": 'half_cover.jpeg',  # Use the uploaded image as the main image
    "template_image_path": 'template2.jpeg',  # Replace with the actual template path
    "change_in_blue_thresthold": 3,
    "sunflower_template": 'flower_template.jpeg'
}


## in cm
arm_init_config = {
  "st1": 2.8,
  "st2": 5.9,
  "st3": 6.7,
  "st4": 3.75,
  "st-tta": 70,
  "st-lmb": -135,
  "st-phi": 65,
  "delta-height-scaling": 0.1,
  "minimum-step-size": 0.1,
  "alignment_depth_threshold": .5
}


CableCar_config = {
   "len_of_thread": 100,
   "num_flowers": 3,
   "micro_adjustment": 3
}

class CableCar:
    def move_to_pos(self, position_to_move, current_position):
        distance_per_position = CableCar_config["len_of_thread"] / CableCar_config["num_flowers"]
        distance_to_move_from_origin = distance_per_position * position_to_move
        distance_to_move = distance_to_move_from_origin - ( current_position * distance_per_position )
        self.move_distance(distance_to_move)
        return distance_to_move_from_origin

    def move_distance(self, distance):  # positive / negative SI unit distance
        return None

class ArmMotor():
  def motion(motor1, motor2, motor3):
    time.sleep(0.1)
    return None
  def depth(): return None

class Sensors():
   def get_camera_frame(): return None
   def get_depth(): return None



class ArmTrajectory():
  def trajectory_small_step(target_height, config): 
    current_height = arm_init_config["st1"] + arm_init_config["st4"] +  arm_init_config["st2"]*math.cos(config["st-tta"]) + arm_init_config["st3"]*math.cos(config["st-phi"])
    delta_height = target_height - current_height
    scaling = delta_height * arm_init_config["delta-height-scaling"]
    if delta_height > arm_init_config["alignment_depth_threshold"]:
      if scaling>0.1:
        step_size = scaling
      else:
        step_size = 0.1
      new_config = config
      desired_height = target_height - arm_init_config["st1"] - arm_init_config["st4"]
      desired_height_for_current_iteration = desired_height * scaling
      while True:
        new_config["st-tta"]+=step_size
        new_config["st-phi"]+=step_size
        new_config["st-lmb"]-=2*step_size
        if arm_init_config["st2"]*math.cos(new_config["st-tta"]) + arm_init_config["st3"]*math.cos(new_config["st-phi"]) - desired_height_for_current_iteration > 0:
          break

    elif delta_height < 0:
      print("ERROR NO SENSE")
    
    else:
      return [True]

    return new_config

  def retract_arm(restart=False):
    target_config = arm_init_config
    ArmMotor.motion(target_config["st-tta"], target_config["st-lmb"], target_config["st-phi"])
    if restart: ArmTrajectory.main()
    

  def main(first_ratio):
    current_config = arm_init_config
    while True:
      current_height = arm_init_config["st1"] + arm_init_config["st4"] +  arm_init_config["st2"]*math.cos(current_config["st-tta"]) + arm_init_config["st3"]*math.cos(current_config["st-phi"])
      target_config = ArmTrajectory.trajectory_small_step(Sensors.get_depth() + current_height,  current_config)
      ArmMotor.motion(target_config["st-tta"], target_config["st-lmb"], target_config["st-phi"])
      current_config = target_config    
      if current_config == [True]:
        print("we're touching sunflower according to depth")
        break
      

    camera_frame = Sensors.get_camera_frame()


    main_image, isolated_pixels, anther_lower_left, anther_width, anther_height = get_box(camera_frame, config["sunflower_template"])
    total_pixels, blue_percentage, red_percentage, green_percentage = give_pixel_percentage(isolated_pixels)
    current_ratio = {"blue": blue_percentage, "non-blue": red_percentage+green_percentage}
    chck = change_in_blue(first_ratio, current_ratio)
    if chck:
      ArmTrajectory.retract_arm(restart=False)
    else:
      ArmTrajectory.retract_arm(restart=True)

    return True





def scheduler():
    with open('data.json') as f:
        data = json.load(f)
    
    flower_locations = data['flowers']
    total_locations = len(flower_locations)
    visited = set()
    loc = None
    
    current_position = 0

    while len(visited) < total_locations:
        # Picking up pollen
        f1 = random.randint(0, total_locations - 1)
        if f1 not in visited:
            visited.add(f1)
            loc = flower_locations[f1]
            print(f"Picking up pollen at location: {loc}")

            CableCar.move_to_pos(f1, current_position)
            current_position = f1
            while True:
                camera_frame = Sensors.get_camera_frame()
                sunflower_main_image, sunflower_isolated_pixels, flower_lower_left, sunflower_width, sunflower_height = get_box(camera_frame, "flower_template.jpeg")
                p1, p2, p3, p4 = coordinates_from_lower_left_width_and_height(flower_lower_left, sunflower_width, sunflower_height)
                if centre_align(p1,p2,p3,p4,camera_frame.shape[1], camera_frame.shape[0]): break
            main_image, isolated_pixels, anther_lower_left, anther_width, anther_height = get_box(camera_frame, "template2.jpeg")
            total_pixels, blue_percentage, red_percentage, green_percentage = give_pixel_percentage(isolated_pixels)
            first_ratio = {"blue": blue_percentage, "non-blue": red_percentage+green_percentage}
            ArmTrajectory.main(first_ratio)

    make_call_via_twilio()

    print("All flowers have been pollinated.")



def centre_align(p1, p2, p3, p4, image_width, image_height):
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
            CableCar.move_distance( CableCar_config["micro_adjustment"])
            return False
        elif dist_y > 0:  # Sunflower is below the center
            CableCar.move_distance( -CableCar_config["micro_adjustment"])
            return False
    else:
        print("Sunflower is centered. No movement required.")
        return True



# Load the main image and template image


def round_to_rgb(pixel):
    r, g, b = pixel
    if r > g and r > b:
        return [255, 0, 0]  # Red
    elif g > r and g > b:
        return [0, 255, 0]  # Green
    elif b > r and b > g:
        return [0, 0, 255]  # Blue
    else:
        if g >= r and g >= b:
            return [0, 255, 0]
        elif r >= g and r >= b:
            return [255, 0, 0]
        else:
            return [0, 0, 255]



def get_box(main_image, template_image_path):
    #main_image = cv2.imread(main_image_path)
    template_image = cv2.imread(template_image_path)

    # Perform template matching
    method = cv2.TM_CCOEFF_NORMED
    result = cv2.matchTemplate(main_image, template_image, method)
    print("LOLZ", result)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print("LMAOZ", min_val, max_val, max_loc)

    # Define the bounding box around the matched region
    top_left = max_loc
    bottom_right = (top_left[0] + template_image.shape[1], top_left[1] + template_image.shape[0])


    isolated_pixels = main_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    return main_image, isolated_pixels, top_left, bottom_right[0] - top_left[0],  bottom_right[1] - top_left[1]


def give_pixel_percentage(main_image):

    # Apply the rounding function to the isolated pixels
    rounded_pixels = np.apply_along_axis(round_to_rgb, 2, main_image).astype(np.uint8)

    # Calculate the percentage of blue, red, and green content
    total_pixels = rounded_pixels.shape[0] * rounded_pixels.shape[1]
    blue_pixels = np.sum(np.all(rounded_pixels == [255, 0, 0], axis=2))
    red_pixels = np.sum(np.all(rounded_pixels == [0, 0, 255], axis=2))
    green_pixels = np.sum(np.all(rounded_pixels == [0, 255, 0], axis=2))

    blue_percentage = (blue_pixels / total_pixels) * 100 if total_pixels != 0 else 0
    red_percentage = (red_pixels / total_pixels) * 100 if total_pixels != 0 else 0
    green_percentage = (green_pixels / total_pixels) * 100 if total_pixels != 0 else 0

    # Print the percentage of each color for debugging
    print(f"Total Pixels: {total_pixels:.2f}")
    print(f"Blue Pixels: {blue_pixels:.2f}")
    print(f"Red Pixels: {red_pixels:.2f}")
    print(f"Green Pixels: {green_pixels:.2f}")

    print(f"Percentage of blue content: {blue_percentage:.2f}%")
    print(f"Percentage of red content: {red_percentage:.2f}%")
    print(f"Percentage of green content: {green_percentage:.2f}%")

    return total_pixels, blue_percentage, red_percentage, green_percentage


def change_in_blue(ratio_first_picture, ratio_current):
    if ( ratio_current["blue"] / ratio_first_picture["blue"] ) > matching_config["change_in_blue_thresthold"]:
        return True
    else: return False


def coordinates_from_lower_left_width_and_height(lower_left, width, height):
    lower_left_x, lower_left_y = lower_left
    upper_left = (lower_left_x, lower_left_y + height)
    lower_right = (lower_left_x + width, lower_left_y)
    upper_right = (lower_left_x + width, lower_left_y + height)
    
    return lower_left, upper_left, lower_right, upper_right


## TODO
#def cordinates_from_lower_left_width_and_height(lower_left, width, height):
    #upper_left = (lower_left[0] + height), (lower_left)


#main_image, isolated_pixels, anther_lower_left, anther_width, anther_height = get_box(config["main_image_path"], config["template_image_path"])
#plt.imshow(cv2.cvtColor(isolated_pixels, cv2.COLOR_BGR2RGB))
#plt.show()


#sunflower_main_image, sunflower_isolated_pixels, flower_lower_left, sunflower_width, sunflower_height = get_box(config["main_image_path"], config["sunflower_template"])
#plt.imshow(cv2.cvtColor(sunflower_isolated_pixels, cv2.COLOR_BGR2RGB))
#plt.show()



#total_pixels, blue_percentage, red_percentage, green_percentage = give_pixel_percentage(isolated_pixels)

## for low detection accuracy, pls use 2 templates to improve accuracy

## ratio is defined like this:
#ratio = {"blue": blue_percentage, "non-blue": red_percentage+green_percentage}
## basically, after first detection, log ratio value and at each point in time log ratio value... then run change_in_blue, if new blue percent / first blue percent is more than a threshold, ie blue has signficantly increased: we say blue is detected



# Display the result
#plt.subplot(1, 2, 1)
#plt.imshow(cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB))
#plt.gca().add_patch(plt.Rectangle(anther_lower_left, anther_width, anther_height, edgecolor='red', facecolor='none', linewidth=2))
#plt.title("Template Matching Result")



#plt.show()

#Motion Of Arm - going down


#Motion Of Arm - going up

from twilio.rest import Client

# Twilio credentials
ACCOUNT_SID = 'AC0bac3c2877518b96bb680d39b1d4b988'
AUTH_TOKEN = 'e37ee8bb97444ea909ba3d2a9134f99e'
FROM_NUMBER = '+19048440281'

def make_call_via_twilio(to_number, message_url):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    call = client.calls.create(
        to=to_number,
        from_=FROM_NUMBER,
        url=message_url
    )
    print("Call initiated successfully!" if call.sid else "Failed to initiate call")

# Example usage
to_number = '+918860069924'  # Recipient's phone number with country code (India)
message_url = 'https://github.com/raghavmukhija/RoboPollinator/blob/main/voice.xml'  # URL to TwiML instructions for the call
make_call_via_twilio(to_number, message_url)



