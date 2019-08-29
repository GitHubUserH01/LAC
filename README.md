# LAC

### Modules

- Motion_prediction: This contains the model trained by W-IL
- Car_following: This contains the model trained by MA-Seq2Seq (Please first unzip the file all_model.ckpt.meta.zip)

## To test the lane change assistance:
cd ./ 

python3 main.py

## To test W-IL
cd ./W_IL 

python3 main.py

## To test MA-Seq2Seq:
cd ./MA_Seq2Seq 

python3 TDpredict_GM.py


