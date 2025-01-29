import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from constant_values import con

# matplotlib.use("TkAgg")  

class CPPN:
    def __init__(self, copy_from=None):
        if copy_from is None:
            self.layers = []
            # 初期化: ランダムなネットワーク構造を生成
            self.layers.append({
                'weights': np.random.randn(2, 5) * 0.1,
                'bias': np.random.randn(5) * 0.1,
                'activation': np.random.choice(['tanh', 'sigmoid', 'relu'])
            })
            self.layers.append({
                'weights': np.random.randn(5, 5) * 0.1,
                'bias': np.random.randn(5) * 0.1,
                'activation': np.random.choice(['tanh', 'sigmoid', 'relu'])
            })
            self.layers.append({
                'weights': np.random.randn(5, 3) * 0.1,
                'bias': np.random.randn(3) * 0.1,
                'activation': 'sigmoid'
            })
        else:
            # コピーコンストラクタ
            self.layers = []
            for layer in copy_from.layers:
                self.layers.append({
                    'weights': layer['weights'].copy(),
                    'bias': layer['bias'].copy(),
                    'activation': layer['activation']
                })

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """ネットワークの重みとバイアスを変異させる"""
        for layer in self.layers:
            # 重みの変異
            mask = np.random.rand(*layer['weights'].shape) < mutation_rate
            layer['weights'] += np.random.randn(*layer['weights'].shape) * mutation_scale * mask
            
            # バイアスの変異
            mask = np.random.rand(*layer['bias'].shape) < mutation_rate
            layer['bias'] += np.random.randn(*layer['bias'].shape) * mutation_scale * mask

def generate_image(cppn, width=64, height=64):
    """CPPNから画像を生成"""
    x = np.tanh(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    inputs = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    
    current = inputs
    for layer in cppn.layers:
        current = np.dot(current, layer['weights']) + layer['bias']
        activation = layer['activation']
        if activation == 'tanh':
            current = np.tanh(current)
        elif activation == 'sigmoid':
            current = 1 / (1 + np.exp(-current))
        elif activation == 'relu':
            current = np.maximum(0, current)
    
    image = current.reshape(height, width, 3)
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)

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

def get_user_selection(population_size):
    """安全なユーザー入力処理"""
    while True:
        try:
            selected = input(f"選択したい画像の番号をカンマ区切りで入力 (0-{population_size-1}): ").strip()
            if not selected:
                raise ValueError("入力が空です")
                
            indices = list(map(int, selected.split(',')))
            if all(0 <= i < population_size for i in indices):
                return indices
                
            print(f"0から{population_size-1}の範囲で入力してください")
            
        except ValueError as e:
            print(f"無効な入力です: {e}")
        except KeyboardInterrupt:
            print("\nプログラムを終了します")
            exit(0)

def create_next_generation(selected_cppns, population_size):
    """選択されたCPPNから次世代を生成"""
    next_gen = []
    num_selected = len(selected_cppns)
    for i in range(population_size):
        parent = selected_cppns[i % num_selected]
        child = CPPN(copy_from=parent)
        child.mutate()
        next_gen.append(child)
    return next_gen

def main():
    population = [CPPN() for _ in range(con.EVOLUTION_POPULATION_SIZE)]
    
    try:
        for gen in range(con.EVOLUTION_GENERATIONS):
            print(f"\nGeneration {gen + 1}/{con.EVOLUTION_GENERATIONS}")
            
            # 画像生成と表示
            images = [generate_image(cppn) for cppn in population]
            display_images(images)
            
            # ユーザー選択
            selected_indices = get_user_selection(con.EVOLUTION_POPULATION_SIZE)
            selected_cppns = [population[i] for i in selected_indices]
            
            # 次世代生成
            population = create_next_generation(selected_cppns, con.EVOLUTION_POPULATION_SIZE)
            
            # 前の画像をクリア
            plt.close('all')
            
        print("\n進化が正常に完了しました")
        
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました")
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()