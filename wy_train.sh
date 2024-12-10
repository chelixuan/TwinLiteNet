echo "---------------------------------------------------------------------"
echo "20241210 tianjin: mean(focal + bce) + 10.0 * tversky_loss \n"
echo "---------------------------------------------------------------------"
python train.py --max_epochs 500 --num_workers 16 --batch_size 32 \
                --height 360 --width 640 \
                --pixel_error_loss mean \
                --alpha 1.0 --beta 10.0 \
                --train /home/chelx/dataset/seg_images/images/train/train_batch_00_tianjin_202408/ \
                --validation False \
                --pretrained /home/chelx/semantic_segmentation/TwinLiteNet/pretrained/best.pth \
                --savedir /home/chelx/ckpt/TwinLetNet/single_head/1210_TJ_1.0mean_10.0skyl
wait

echo "---------------------------------------------------------------------"
echo "20241210 tianjin+lyon : mean(focal + bce) + 10.0 * tversky_loss \n"
echo "---------------------------------------------------------------------"
python train.py --max_epochs 500 --num_workers 16 --batch_size 32 \
                --height 360 --width 640 \
                --pixel_error_loss mean \
                --alpha 1.0 --beta 10.0 \
                --train /home/chelx/dataset/seg_images/images/train/ \
                --val /home/chelx/dataset/seg_images/images/val/ \
                --pretrained /home/chelx/semantic_segmentation/TwinLiteNet/pretrained/best.pth \
                --savedir /home/chelx/ckpt/TwinLetNet/single_head/1210_TJ+Lyon_1.0mean_10.0skyl
wait

echo "\nTHE SYSTEM WILL BE SHUTDOWN NOW !!! \n"
shutdown -h now
