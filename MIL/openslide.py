import slideio
import matplotlib.pyplot as plt

slide = slideio.open_slide('/home/dsi/zurkin/data/27124.svs', 'SVS')
num_scenes = slide.num_scenes
scene = slide.get_scene(0)
print(num_scenes, scene.name, scene.rect, scene.num_channels)
raw_string = slide.raw_metadata
print(raw_string.split("|"))

for channel in range(scene.num_channels):
    print(scene.get_channel_data_type(channel))

image = scene.read_block(size=(500,0))
#plt.imshow(image)
print(image.shape)
