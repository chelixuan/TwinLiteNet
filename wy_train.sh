echo "1010 Lyon2024 : \n"
python train.py --max_epochs 500 --num_workers 12 --batch_size 1 \
                --height 720 --width 1280 \
                --pretrained /home/chelx/semantic_segmentation/TwinLiteNet/pretrained/best.pth \
                --savedir /home/chelx/ckpt/TwinLetNet/lyon2024/single_head/base_bce_500epochs
wait

# echo "\nTHE SYSTEM WILL BE SHUTDOWN NOW !!! \n"
# shutdown -h now
