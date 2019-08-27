# LAC

### Running

## To test the lane change assistance:
cd ./ 

python3 main.py

## To test W-IL
cd ./W_IL 

python3 main.py

## To test MA-Seq2Seq:
cd ./MA_Seq2Seq 

python3 main.py

### Modules

- dynamics.py: This contains code for car dynamics.
- car.py: Relevant code for different car models (human-driven, autonomous, etc.)
- feature.py: Definition of features.
- lane.py: Definition of driving lanes.
- trajectory.py: Definition of trajectories.
- world.py: This code contains different scenarios (each consisting of lanes/cars/etc.).
- visualize.py: This contains the code for visualization (GUI).
