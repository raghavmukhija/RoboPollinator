import json
import random

def scheduler():
    with open('Competition\data.json') as f:
        data = json.load(f)
    
    flower_locations = data['flowers']
    total_locations = len(flower_locations)
    visited = set()
    loc = None
    
    while len(visited) < total_locations:
        # Picking up pollen
        while True:
            f1 = random.randint(0, total_locations - 1)
            if f1 not in visited:
                visited.add(f1)
                loc = flower_locations[f1]
                print(f"Picking up pollen at location: {loc}")
                break
        
        # Dropping the pollen
        while True:
            f2 = random.randint(0, total_locations - 1)
            if f2 not in visited:
                visited.add(f2)
                loc = flower_locations[f2]
                print(f"Dropping pollen at location: {loc}")
                break
    
    print("All flowers have been pollinated.")

# Example call to the function (Uncomment to run)
scheduler()
