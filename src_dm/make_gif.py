import imageio
import os

image_folder = '/Users/gufran/Developer/Projects/AI/CancerCellDataAugmentation/src/results/DDPM_Uncondtional'

output_gif = '/Users/gufran/Developer/Projects/AI/CancerCellDataAugmentation/outputffff.gif'

images = []
image_names = [f"{i}.jpg" for i in range(218)]
for filename in image_names:
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(image_folder, filename)
        images.append(imageio.imread(img_path))


frame_duration = 0.001  # 0.1 seconds between frames
imageio.mimsave(output_gif, images, duration=frame_duration)