# echo "Run on thres = 0.0"
# bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_uvc_prop_thres_0 --visualize --prop_thres 0.0

# echo "Run on thres = 0.1"
# bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_uvc_prop_thres_0_1 --visualize --prop_thres 0.1

# echo "Run on thres = 0.2"
# bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_uvc_prop_thres_0_2 --visualize --prop_thres 0.2

echo "Run on thres = 0.3"
bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_prop_dino_thres_0_3 --visualize --prop_thres 0.3 --temp 0.001 

echo "Run on thres = 0.4"
bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_prop_dino_thres_0_4 --visualize --prop_thres 0.4 --temp 0.001 

echo "Run on thres = 0.5"
bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_prop_dino_thres_0_5 --visualize --prop_thres 0.5 --temp 0.001 

echo "Run on thres = 0.6"
bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_prop_dino_thres_0_6 --visualize --prop_thres 0.6 --temp 0.001 

echo "Run on thres = 0.7"
bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_prop_dino_thres_0_7 --visualize --prop_thres 0.7 --temp 0.001 

echo "Run on thres = 0.8"
bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_prop_dino_thres_0_8 --visualize --prop_thres 0.8 --temp 0.001 

echo "Run on thres = 0.9"
bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_prop_dino_thres_0_9 --visualize --prop_thres 0.9 --temp 0.001 


# echo "Run on thres = 0.4 and temp = 0.5"
# bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_prop_dino_thres_0_4_temp_0_5 --visualize --prop_thres 0.4 --temp 0.5

# echo "Run on thres = 0.4 and temp = 0.1"
# bash ./scripts/online_davis_sam_prop_dino.sh ./outputs_sam_prop_dino_thres_0_4_temp_0_1 --visualize --prop_thres 0.5 --temp 0.1
