import yaml

class Config:
    """設定情報を管理するクラス"""

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        # 各設定をクラス属性として登録
        self.EVOLUTION_GENERATIONS = config["evolution"]["generations"]
        self.EVOLUTION_POPULATION_SIZE = config["evolution"]["population_size"]

        self.CPPN_LAYER_SIZES = config["cppn"]["layer_sizes"]
        self.CPPN_MAX_LAYER_SIZES = config["cppn"]["max_layer_size"]
        self.CPPN_ACTIVATIONS = config["cppn"]["activations"]
        self.CPPN_OUTPUT_ACTIVATION = config["cppn"]["output_activation"]

        self.MUTATION_RATE = config["mutation"]["mutation_rate"]
        self.MUTATION_SCALE = config["mutation"]["mutation_scale"]

        self.IMAGE_WIDTH = config["image"]["width"]
        self.IMAGE_HEIGHT = config["image"]["height"]

# 設定ファイルを読み込み、Configクラスのインスタンスを生成
con = Config("cppn.yaml")
