# camera_shake

usage:
1) use inpainting, copymoves, inpaint in resolution 200x200, scale original moves by 2  
    ```python apply_camera_shake.py --input_video /mnt/f/works/test_task_videos/01.mp4 --output_video /mnt/f/works/test.mp4 --inpaint --scale 2 --copymoves --inpaint_width 200 --inpaint_height 200```

2) use bordercopy, copymoves, no inpainting  
   ```python apply_camera_shake.py --input_video /mnt/f/works/test_task_videos/01.mp4 --output_video /mnt/f/works/test.mp4 --scale 2 --copymoves```
