import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

RESULT_PATH = './results.csv'
COLOR_PATH = './dataset/Color'
GRAY_PATH = './dataset/Gray'

REMAIN_KEY = 'k'
REMOVE_KEY = 'l'
UNDO_KEY = 'z'


def main():
    df = pd.read_csv(RESULT_PATH)

    if not 'result' in df.columns:
        df['result'] = 0

    start_index = int(input('start index: '))
    end_index = int(input('end index: '))

    for i in range(start_index, min(end_index, len(df))):
        def on_key_press(event):
            if event.key == REMAIN_KEY:
                df['result'][i] = 1
                plt.close()
            elif event.key == REMOVE_KEY:
                df['result'][i] = 0 
                plt.close()
            elif event.key == UNDO_KEY:
                df['result'][i - 1] = 0
            else:
                return

        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', on_key_press)

        label = df['label'][i]

        color_image = cv2.imread(get_color_path(label))
        gray_image = cv2.imread(get_gray_path(label))

        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

        color_image = cv2.resize(color_image, gray_image.shape)

        (_, diff) = ssim(color_image, gray_image, full=True)

        plt.subplot(131)
        plt.xlabel('COLOR IMAGE')
        plt.imshow(color_image, cmap='gray')

        plt.subplot(132)
        plt.xlabel('GRAY IMAGE')
        plt.imshow(gray_image, cmap='gray')

        plt.subplot(133)
        plt.xlabel(f'DIFF (INTENSITY={np.sum(diff):.0f})')
        plt.imshow(diff, cmap='gray')

        plt.title(f"SSIM={df['ssim'][i]:.1f} | PSNR={df['psnr'][i]:.1f}")

        plt.show()

        # Checkpointing
        if i % 10 == 0:
            df.to_csv(RESULT_PATH, index=False)


def get_color_path(label: str) -> str:
    return f"{COLOR_PATH}/{label}_4.png"


def get_gray_path(label: str) -> str:
    return f"{GRAY_PATH}/{label}_2.jpg"



if __name__=='__main__':
    main()
