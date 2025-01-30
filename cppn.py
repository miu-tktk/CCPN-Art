import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from constant_values import con
import matplotlib
import tkinter as tk
matplotlib.use('TkAgg')


class CPPN:
    def __init__(self, copy_from=None):
        self.innovations = {}  # イノベーション番号管理
        self.connections = []  # 接続エッジリスト
        self.layers = []       # レイヤー情報

        if copy_from is None:
            self._initialize_new_network()
        else:
            self._copy_existing_network(copy_from)

    def _initialize_new_network(self):

        self._add_layer(2, 16, 'hidden', activation='sine')  # ノード数を16に
        self._add_layer(16, 12, 'hidden', activation='tanh')
        self._add_layer(12, 3, 'output', activation='sigmoid')
        self._update_connections()

    def _copy_existing_network(self, source):
        self.innovations = source.innovations.copy()
        self.connections = source.connections.copy()
        self.layers = [{
            'input_size': l['input_size'],
            'output_size': l['output_size'],
            'weights': l['weights'].copy(),
            'bias': l['bias'].copy(),
            'activation': l['activation'],
            'type': l['type'],
            'id': l['id']
        } for l in source.layers]

    def _add_layer(self, in_size, out_size, layer_type, activation):
        layer_id = max([l['id'] for l in self.layers], default=-1) + 1
        self.layers.append({
            'id': layer_id,
            'input_size': in_size,
            'output_size': out_size,
            'weights': np.random.randn(in_size, out_size) * 2.0,
            'bias': np.random.randn(out_size) * 0.5,
            'activation': activation,
            'type': layer_type
        })
        self._update_connections()

    def _update_connections(self):
        self.connections = []
        sorted_layers = sorted(self.layers, key=lambda x: x['id'])
        for i in range(len(sorted_layers)-1):
            self.connections.append((sorted_layers[i]['id'], sorted_layers[i+1]['id']))
        self._ensure_dag()

    def _ensure_dag(self):
        # networkxを用いてDAGかどうかをチェック
        G = nx.DiGraph()
        G.add_edges_from(self.connections)
        if not nx.is_directed_acyclic_graph(G):
            raise RuntimeError("Non-DAG structure detected")

    def mutate(self, mutation_rate=0.1):
        mutation_type = np.random.choice([
            'weight', 'activation', 'add_layer', 'remove_layer', 'add_connection'
        ], p=[0.4, 0.2, 0.15, 0.15, 0.1])

        if mutation_type == 'weight':
            self._mutate_weights(mutation_rate)
        elif mutation_type == 'activation':
            self._mutate_activation()
        elif mutation_type == 'add_layer' and self.connections:  # 接続がある場合のみレイヤー追加
            self._add_random_layer()
        elif mutation_type == 'remove_layer':
            self._remove_random_layer()
        elif mutation_type == 'add_connection':
            self._add_random_connection()

    def _mutate_weights(self, rate):
        for layer in self.layers:
            if layer['type'] == 'input':
                continue
            mutation = np.random.randn(*layer['weights'].shape) * rate
            layer['weights'] += mutation
            layer['bias'] += np.random.randn(*layer['bias'].shape) * rate

    def _mutate_activation(self):
        for layer in self.layers:
            if layer['type'] in ['input', 'output']:
                continue
            layer['activation'] = np.random.choice(['tanh', 'sigmoid', 'sine'])

    def _add_random_layer(self):
        if len(self.layers) >= con.CPPN_MAX_LAYER_SIZES:
            return

        # 既存の接続をランダムに選択
        if not self.connections:
            return  # 接続がない場合は何もしない

        u, v = self.connections[np.random.randint(len(self.connections))]
        u_layer = next(l for l in self.layers if l['id'] == u)
        v_layer = next(l for l in self.layers if l['id'] == v)

        # 新しいレイヤーの追加
        new_size = np.random.randint(3, 8)
        self._add_layer(u_layer['output_size'], new_size, 'hidden', np.random.choice(['tanh', 'sigmoid', 'sine']))
        new_layer = self.layers[-1]

        # 接続の更新
        if (u, v) in self.connections:  # 接続が存在する場合のみ削除
            self.connections.remove((u, v))
        self.connections.append((u, new_layer['id']))
        self.connections.append((new_layer['id'], v))

        # 重みの調整
        v_layer['input_size'] = new_size
        v_layer['weights'] = np.random.randn(new_size, v_layer['output_size']) * 0.1

    def _remove_random_layer(self):
        if len(self.layers) <= 3:
            return

        # 削除可能なレイヤーを選択
        candidates = [l for l in self.layers if l['type'] == 'hidden']
        if not candidates:
            return

        target = np.random.choice(candidates)
        incoming = [c for c in self.connections if c[1] == target['id']]
        outgoing = [c for c in self.connections if c[0] == target['id']]

        # 接続の再編成
        for u, _ in incoming:
            for _, v in outgoing:
                if (u, v) not in self.connections:
                    self.connections.append((u, v))

        # レイヤー削除
        self.layers = [l for l in self.layers if l['id'] != target['id']]
        self._update_connections()

    def _add_random_connection(self):
        # 可能な新しい接続を探索
        possible_connections = []
        for i, u in enumerate(self.layers):
            for j, v in enumerate(self.layers):
                if i >= j:
                    continue
                if (u['id'], v['id']) not in self.connections:
                    possible_connections.append((u['id'], v['id']))

        if possible_connections:
            new_conn = possible_connections[np.random.randint(len(possible_connections))]
            self.connections.append(new_conn)
            self._ensure_dag()


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))*100


def sine(x):
    return np.sin(x)


def tanh(x):
    return np.tanh(x)


def generate_image(cppn, width=64, height=64):
    """CPPNから画像を生成"""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    inputs = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    # ここまで
    current = inputs
    for layer in sorted(cppn.layers, key=lambda l: l['type'] != 'input'):
        if layer['type'] == 'input':
            continue

        current = np.dot(current, layer['weights']) + layer['bias']
        activation = layer['activation']
        if activation == 'tanh':
            current = tanh(current)
        elif activation == 'sigmoid':
            current = sigmoid(current)
        elif activation == 'sine':
            current = sine(current)
    # 出力データの形状を確認
    print(f"Final output shape: {current.shape}")
    image = current.reshape(height, width, 3)
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def display_images(favored_images, unfavored_images):

    """favored と unfavored の画像を横に並べて表示"""
    
    # 画像の最大数を取得（favored, unfavored のどちらか多い方）
    n_favored = len(favored_images)
    n_unfavored = len(unfavored_images)
    n = con.EVOLUTION_POPULATION_SIZE

    # # 画像の表示レイアウトを決定
    rows = 1 
    cols = con.EVOLUTION_POPULATION_SIZE

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # ウィンドウの位置を固定
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを隠す
    # ウィンドウの位置を固定
    fig.canvas.manager.window.wm_geometry("640x550+40+10")

    # 1行1列の場合の調整
    if rows == 1 and cols == 2:
        axes = np.array([[axes]])  
    axes = np.atleast_2d(axes)  # 2D配列として統一

    # Favored（左側）
    for idx, (ax, img) in enumerate(zip(axes.flat[:n_favored], favored_images), start=1):
        ax.imshow(img)
        ax.axis('off')
        ax.text(5, 5, f"F{idx}", color="white", fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # Unfavored（右側）
    for idx, (ax, img) in enumerate(zip(axes.flat[n_favored:], unfavored_images), start=1):
        ax.imshow(img)
        ax.axis('off')
        ax.text(5, 5, f"U{idx}", color="white", fontsize=12, 
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

def get_user_favor_selection(population_size):
    """安全なユーザー入力処理"""
    while True:
        try:
            selected = input(f"一番好きな画像の番号を入力 (1-{int(population_size / 2)}): ").strip()
            if not selected or int(selected) > int(population_size / 2):
                raise ValueError(f"入力が空もしくは適切に選択されていません。1から{int(population_size / 2)}の間で選択してください。")

            indices = list(map(int, selected.split(',')))
            if all(0 <= i < population_size for i in indices):
                # return indices, selected
                return indices

            print(f"1から{population_size / 2}の範囲で入力してください")

        except ValueError as e:
            print(f"無効な入力です: {e}")
        except KeyboardInterrupt:
            print("\nプログラムを終了します")
            exit(0)


def create_next_generation(selected_cppns, population_size):
    """選択されたCPPNから次世代を生成 (DAGであることを保証)"""
    next_gen = []
    num_selected = len(selected_cppns)
    max_attempts = 50  # 1個体を生成するのに最大で50回リトライする例

    i = 0
    while len(next_gen) < population_size:
        parent = selected_cppns[i % num_selected]

        # 特定の親から子を作成することを複数回リトライ
        for attempt in range(max_attempts):
            child = CPPN(copy_from=parent)
            child.mutate()

            # DAGチェック
            try:
                child._ensure_dag()  # networkx等でDAGかどうかをチェック
            except RuntimeError:
                # Non-DAG だった場合は再生成
                continue
            else:
                # DAGであれば次世代に採用してリトライを打ち切る
                next_gen.append(child)
                break

        else:
            # for-else文：max_attempts回リトライしてもDAGが作れない場合
            raise RuntimeError("Valid DAG child could not be generated after multiple attempts.")

        i += 1

    return next_gen


def main():
    population = [CPPN() for _ in range(con.EVOLUTION_POPULATION_SIZE)]
    try:
        for gen in range(con.EVOLUTION_GENERATIONS):
            print(f"\nGeneration {gen + 1}/{con.EVOLUTION_GENERATIONS}")
            unselected_cppn_history = []
            if gen == 0:
                favored_images = [generate_image(cppn) for cppn in population][:2]
                unfavored_images = [generate_image(cppn) for cppn in population][2:]
                display_images(favored_images, unfavored_images)

            else:
                # 画像生成と表示
                favored_images = [generate_image(cppn) for cppn in favored_population]
                unfavored_images = [generate_image(cppn) for cppn in unfavored_population]
                display_images(favored_images, unfavored_images)

            # ユーザー選択
            selected_indices = get_user_favor_selection(con.EVOLUTION_POPULATION_SIZE)
            selected_cppns = [population[i] for i in selected_indices]
            unselected_cppns = [item for item in population if item not in selected_cppns]
            print(unselected_cppns)
            unselected_cppn_history.append(unselected_cppns[0])
            favored_population = create_next_generation(selected_cppns, int(con.EVOLUTION_POPULATION_SIZE / 2))
            if gen == 0:
                print(unselected_cppn_history)
                unfavored_population = create_next_generation(unselected_cppn_history, int(con.EVOLUTION_POPULATION_SIZE / 2))
            else:
                unfavored_population = create_next_generation(unselected_cppn_history[:gen], int(con.EVOLUTION_POPULATION_SIZE / 2))
            # 前の画像をクリア
            plt.close('all')

        print("\n進化が正常に完了しました")

    except KeyboardInterrupt:
        print("\nユーザーによって中断されました")
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()
