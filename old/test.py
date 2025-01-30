import pygame
import random
import math
import time
import numpy as np

# 定数の設定
WIDTH, HEIGHT = 800, 600
FPS = 60
BLACK = (0, 0, 0)
MUTATION_RATE = 0.1
MAX_SPEED = 3  # 追加: 最大速度制限

# 粒子数
PARTICLE_COUNT_POSITIVE = 1000
PARTICLE_COUNT_NEGATIVE = 1000

class Particle:
    def __init__(self, x, y, charge, size, speed, attraction_strength):
        self.x = x
        self.y = y
        self.charge = charge
        self.size = size
        self.speed = speed
        self.attraction_strength = attraction_strength
        self.creation_time = time.time()
        self.lifespan = 0  # 生存時間追跡用
        self.vx = random.uniform(-self.speed, self.speed)
        self.vy = random.uniform(-self.speed, self.speed)

    def apply_force(self, fx, fy):
        # 速度制限を追加
        self.vx = max(-MAX_SPEED, min(MAX_SPEED, self.vx + fx))
        self.vy = max(-MAX_SPEED, min(MAX_SPEED, self.vy + fy))

    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan = time.time() - self.creation_time

        # 壁での跳ね返り（位置補正付き）
        if self.x < self.size:
            self.x = self.size
            self.vx = abs(self.vx)
        elif self.x > WIDTH - self.size:
            self.x = WIDTH - self.size
            self.vx = -abs(self.vx)
        if self.y < self.size:
            self.y = self.size
            self.vy = abs(self.vy)
        elif self.y > HEIGHT - self.size:
            self.y = HEIGHT - self.size
            self.vy = -abs(self.vy)

    def draw(self, screen):
        color = (255, 0, 0) if self.charge > 0 else (0, 0, 255)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

def calculate_force(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    distance = math.hypot(dx, dy)

    if distance == 0:
        return 0, 0

    force_magnitude = (abs(p1.charge) * abs(p2.charge)) * (p1.attraction_strength + p2.attraction_strength)
    force = force_magnitude / (distance ** 2)
    
    fx = force * (dx / distance)
    fy = force * (dy / distance)
    return fx, fy

def check_collision(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y) < (p1.size + p2.size)

def generate_initial_particles():
    particles = []
    for _ in range(PARTICLE_COUNT_POSITIVE):
        particles.append(Particle(
            x=random.randint(20, WIDTH - 20),
            y=random.randint(20, HEIGHT - 20),
            charge=random.randint(1, 20),
            size=random.randint(5, 10),
            speed=random.uniform(1, 3),
            attraction_strength=random.uniform(5, 15)
        ))

    for _ in range(PARTICLE_COUNT_NEGATIVE):
        particles.append(Particle(
            x=random.randint(20, WIDTH - 20),
            y=random.randint(20, HEIGHT - 20),
            charge=random.randint(-20, -1),
            size=random.randint(5, 10),
            speed=random.uniform(1, 3),
            attraction_strength=random.uniform(5, 15)
        ))
    return particles

def reproduce(survivors):
    if not survivors:
        return generate_initial_particles()

    # 適応度（生存時間）でソート
    survivors.sort(key=lambda p: p.lifespan, reverse=True)
    
    new_particles = []
    target_count = PARTICLE_COUNT_POSITIVE + PARTICLE_COUNT_NEGATIVE
    
    for _ in range(target_count):
        parent = random.choice(survivors)
        
        # 突然変異を適用
        new_size = max(5, parent.size * random.uniform(1-MUTATION_RATE, 1+MUTATION_RATE))
        new_speed = max(0.5, parent.speed * random.uniform(1-MUTATION_RATE, 1+MUTATION_RATE))
        new_strength = max(1, parent.attraction_strength * random.uniform(1-MUTATION_RATE, 1+MUTATION_RATE))
        
        new_particles.append(Particle(
            random.randint(20, WIDTH-20),
            random.randint(20, HEIGHT-20),
            parent.charge,
            int(new_size),
            new_speed,
            new_strength
        ))
    return new_particles

def main():
    
    pygame.init()  # Pygameの初期化
    screen = pygame.display.set_mode((WIDTH, HEIGHT))  # ウィンドウの設定
    pygame.display.set_caption("Improved Particle Simulation")  # タイトルバーの設定
    clock = pygame.time.Clock()  # クロックの設定
    particles = generate_initial_particles()  # 初期粒子の生成
    generation = 1  # 世代数

    # メインループ
    running = True
    while running:
        screen.fill(BLACK)  # 画面を黒で塗りつぶす

        # イベント処理
        for event in pygame.event.get():  # イベントを取得
            if event.type == pygame.QUIT:  # 閉じるボタンが押されたら
                running = False  # ループを抜ける

        # 物理計算
        for i, p1 in enumerate(particles):  # 全粒子の組み合わせに対して
            for p2 in particles[i+1:]:  # 重複を避けるため、i+1から開始
                fx, fy = calculate_force(p1, p2)  # 力を計算
                p1.apply_force(fx, fy)  # 力を適用
                p2.apply_force(-fx, -fy)  # 力を逆向きに適用

        # 移動と衝突判定
        to_remove = set()  # 削除対象の粒子インデックス
        for i, p1 in enumerate(particles):  # 全粒子に対して
            p1.move()  # 移動
            for j, p2 in enumerate(particles[i+1:], i+1):
                if check_collision(p1, p2):
                    if np.sign(p1.charge) == np.sign(p2.charge):
                        # 同種電荷: 反発処理
                        dx = p2.x - p1.x
                        dy = p2.y - p1.y
                        distance = math.hypot(dx, dy)
                        if distance == 0:
                            continue
                        
                        # 速度交換と位置補正
                        nx = dx / distance
                        ny = dy / distance
                        overlap = (p1.size + p2.size) - distance
                        
                        p1.x += nx * overlap * abs(p1.charge)
                        p1.y += ny * overlap * abs(p1.charge)
                        p2.x -= nx * overlap * abs(p2.charge)
                        p2.y -= ny * overlap * abs(p2.charge)
                        
                        v1 = (p1.vx * nx + p1.vy * ny)
                        v2 = (p2.vx * nx + p2.vy * ny)
                        new_v1 = v2
                        new_v2 = v1
                        
                        p1.vx += (new_v1 - v1) * nx
                        p1.vy += (new_v1 - v1) * ny
                        p2.vx += (new_v2 - v2) * nx
                        p2.vy += (new_v2 - v2) * ny
                    else:
                        # 異種電荷: 対消滅
                        to_remove.add(i)
                        to_remove.add(j)

        # 粒子削除と世代更新
        particles = [p for idx, p in enumerate(particles) if idx not in to_remove]
        
        # # 4体以下で次世代へ
        # if len(particles) <= 4:
        #     print(f"Generation {generation} ended. Remaining: {len(particles)}")
        #     generation += 1
        #     particles = reproduce(particles)

        # 描画
        for p in particles:
            p.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()