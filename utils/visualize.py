import cv2 as cv
import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 30})

def overlay_masks(input, masks):

    masks = np.dstack(masks)

    colors = [np.flip(np.array(random.choices(range(256), k=3))) for i in range(masks.shape[-1])]
    background = input.copy().astype(np.float)
    H, W, _ = background.shape

    for row in range(H-1):
        for col in range(W-1):
            value = masks[row][col]
            if value.sum() > 1:
                temp = (colors[0]).astype(np.float) *(value[0]).astype(np.float) * (1/value.sum()) 
                for i in range(masks.shape[-1]-1):
                    temp += (colors[i+1]).astype(np.float) *(value[i+1]).astype(np.float) * (1/value.sum()) 
                background[row][col] = temp
            elif value.sum() == 1:
                background[row][col] = colors[np.argwhere(value == 1).item()]
            else:
                pass

    return (input.copy() * 0.25 + background * 0.75).astype(np.uint8)

def display_masks(input, masked):
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(121, title='Cropped Input Image')
    ax2 = fig.add_subplot(122, title='Overlayed Mask')
    ax1.imshow(input)
    ax1.axis('off')
    ax2.imshow(masked)
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig("results.png", bbox_inches='tight', pad_inches=0, dpi=300, transparent=False)
    print("Saved output to results.png!")
    plt.show()