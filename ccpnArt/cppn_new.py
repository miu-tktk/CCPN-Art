import numpy as np
from constant_values import con
import matplotlib.pyplot as plt

class CPPN:
    def __init__(self, x, y, w, b):
        self.x = x
        self.y = y
        self.w = w
        self.b = b

    def calc(self):
        f = lambda x: np.tanh(x)
        output = f(np.dot(self.w, np.array([self.x, self.y, 1]).T) + self.b)
        return output

def display_images(images):
    """非ブロッキングで画像をグリッド表示"""
    n = len(images)
    rows = int(np.ceil(n / 5))
    cols = min(n, 5)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
        
    for idx, (ax, img) in enumerate(zip(axes.flat, images), start=1):
        ax.imshow(img)
        ax.axis('off')
        ax.text(5, 5, str(idx), color="white", fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # 画像を短時間表示

def main():
    initial_population = []
    img = np.zeros((256, 256, 3))
    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    for num in range(con.EVOLUTION_POPULATION_SIZE):
        for i in range(256):
            for j in range(256):
                r = CPPN(x[i], y[j], np.random.randn(3), np.random.randn(1)).calc
                g = CPPN(x[i], y[j], np.random.randn(3), np.random.randn(1)).calc
                b = np.array([0])
                img[i, j] = np.concatenate([r, g, b], axis=0)
        initial_population.append(img)
    
    display_images(initial_population)

if __name__ == "__main__":
    main()



    




