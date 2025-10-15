# Платформер на Pygame: горизонтальный скролл, финиш справа, стомп врагов, шипы и огонь.
# Доработки: двойной прыжок и минимальный зазор между препятствиями.
# Управление: A/D или ←/→ — ходьба, Space — прыжок (до двух), R — рестарт, Esc — выход.

import sys
import pygame

# --- Константы ---
WIDTH, HEIGHT = 960, 540
TILE = 48
FPS = 60

GRAVITY = 0.7
MOVE_SPEED = 5
JUMP_SPEED = 14
STOMP_BOUNCE = 10
ENEMY_SPEED = 2.0

MAX_JUMPS = 2           # всего прыжков «в воздухе»: один с земли + один в воздухе
HAZARD_GAP_TILES = 5    # минимальный зазор (в тайлах) между ловушками

BG_COLOR = (20, 22, 30)
PLATFORM_COLOR = (70, 80, 110)
PLAYER_COLOR = (220, 240, 255)
ENEMY_COLOR = (220, 80, 80)
SPIKES_COLOR = (200, 200, 220)
FIRE_COLOR = (255, 120, 0)
GOAL_COLOR = (255, 220, 0)
TEXT_COLOR = (240, 240, 240)

# --- Вспомогательная расстановка ловушек с зазором ---
def place_hazards_spaced(cols_range, min_gap, pattern=("^", "~", "^")):
    """
    Возвращает список (col, symbol) с гарантированным минимальным интервалом между соседними ловушками.
    """
    placed = []
    last_col = -10**9
    pi = 0
    for c in cols_range:
        if c - last_col >= min_gap:
            placed.append((c, pattern[pi]))
            pi = (pi + 1) % len(pattern)
            last_col = c
    return placed

# --- Генерация широкого уровня ---
def make_level_map(width_tiles=120, height_tiles=11):
    rows = [["." for _ in range(width_tiles)] for _ in range(height_tiles)]
    # Нижняя земля по всей длине
    for x in range(width_tiles):
        rows[height_tiles - 1][x] = "X"

    # Несколько платформ по высоте для геймплея
    for x in range(8, 24):
        rows[height_tiles - 3][x] = "X"
    for x in range(32, 44):
        rows[height_tiles - 5][x] = "X"
    for x in range(52, 64):
        rows[height_tiles - 4][x] = "X"
    for x in range(74, 82):
        rows[height_tiles - 6][x] = "X"
    for x in range(90, 104):
        rows[height_tiles - 3][x] = "X"

    # Старт игрока
    rows[height_tiles - 2][2] = "P"

    # Враги на земле и на платформах
    rows[height_tiles - 2][12] = "E"
    rows[height_tiles - 6][36] = "E"
    rows[height_tiles - 5][58] = "E"
    rows[height_tiles - 7][76] = "E"
    rows[height_tiles - 2][96] = "E"

    # Ловушки на земле с минимальным зазором
    ground_row = height_tiles - 2
    start_c = 14
    end_c = width_tiles - 8
    for c, sym in place_hazards_spaced(range(start_c, end_c), HAZARD_GAP_TILES):
        rows[ground_row][c] = sym

    # Финишный флаг справа
    rows[height_tiles - 2][width_tiles - 3] = "G"

    return ["".join(r) for r in rows]

LEVEL_MAP = make_level_map()
WORLD_W = len(LEVEL_MAP[0]) * TILE
WORLD_H = len(LEVEL_MAP) * TILE

# --- Классы сущностей ---
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((TILE, TILE))
        self.image.fill(PLATFORM_COLOR)
        self.rect = self.image.get_rect(topleft=(x, y))

class Hazard(pygame.sprite.Sprite):
    def __init__(self, x, y, kind):
        super().__init__()
        self.kind = kind  # "spikes" или "fire"
        self.image = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        if kind == "spikes":
            w, h = TILE, TILE
            tri_w = w // 3
            for i in range(3):
                px = i * tri_w
                pygame.draw.polygon(
                    self.image, SPIKES_COLOR,
                    [(px + tri_w // 2, 8), (px + 4, h - 6), (px + tri_w - 4, h - 6)]
                )
        elif kind == "fire":
            pygame.draw.rect(self.image, FIRE_COLOR, (0, TILE // 3, TILE, TILE // 2), border_radius=8)
            pygame.draw.rect(self.image, (255, 180, 0), (6, TILE // 2, TILE - 12, TILE // 2 - 6), border_radius=8)
        self.rect = self.image.get_rect(topleft=(x, y))

class Goal(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        pygame.draw.rect(self.image, (180, 180, 180), (8, 6, 6, TILE - 12), border_radius=2)
        pygame.draw.polygon(self.image, GOAL_COLOR, [(14, 10), (34, 18), (14, 26)])
        self.rect = self.image.get_rect(topleft=(x, y))

class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, platforms):
        super().__init__()
        self.image = pygame.Surface((TILE - 8, TILE - 8))
        self.image.fill(ENEMY_COLOR)
        self.rect = self.image.get_rect(midbottom=(x + TILE // 2, y + TILE))
        self.vx = ENEMY_SPEED
        self.vy = 0.0
        self.platforms = platforms

    def apply_gravity(self):
        self.vy += GRAVITY
        if self.vy > 20:
            self.vy = 20

    # Совместимо с Group.update(args)
    def update(self, *args):
        # Горизонталь
        self.rect.x += int(self.vx)
        hits = pygame.sprite.spritecollide(self, self.platforms, False)
        for h in hits:
            if self.vx > 0:
                self.rect.right = h.rect.left
            elif self.vx < 0:
                self.rect.left = h.rect.right
            self.vx *= -1

        # Вертикаль
        self.apply_gravity()
        self.rect.y += int(self.vy)
        hits = pygame.sprite.spritecollide(self, self.platforms, False)
        for h in hits:
            if self.vy > 0:
                self.rect.bottom = h.rect.top
            elif self.vy < 0:
                self.rect.top = h.rect.bottom
            self.vy = 0

        # Разворот у края
        front_x = self.rect.midbottom[0] + (8 if self.vx > 0 else -8)
        probe_rect = pygame.Rect(front_x - 2, self.rect.bottom + 2, 4, 6)
        grounded_ahead = any(probe_rect.colliderect(p.rect) for p in self.platforms)
        if not grounded_ahead:
            self.vx *= -1

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, platforms, enemies, hazards, goals):
        super().__init__()
        self.image = pygame.Surface((TILE - 10, TILE - 8))
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect(midbottom=(x + TILE // 2, y + TILE))

        self.vx = 0.0
        self.vy = 0.0
        self.platforms = platforms
        self.enemies = enemies
        self.hazards = hazards
        self.goals = goals
        self.on_ground = False
        self.alive = True
        self.won = False

        # Двойной прыжок
        self.jumps_used = 0
        self.jump_was_down = False  # edge-trigger

    def handle_input(self, keys):
        self.vx = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.vx -= MOVE_SPEED
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.vx += MOVE_SPEED

        # Edge-trigger на Space: сработает только при "нажатии", а не при удержании
        jump_down = keys[pygame.K_SPACE]
        if jump_down and not self.jump_was_down:
            # Можно прыгать с земли или сделать дополнительный прыжок в воздухе
            if self.on_ground or self.jumps_used < MAX_JUMPS:
                self.vy = -JUMP_SPEED
                self.on_ground = False
                self.jumps_used += 1
        self.jump_was_down = jump_down

    def apply_gravity(self):
        self.vy += GRAVITY
        if self.vy > 22:
            self.vy = 22

    def move_and_collide(self):
        # Горизонталь
        self.rect.x += int(self.vx)
        hits = pygame.sprite.spritecollide(self, self.platforms, False)
        for h in hits:
            if self.vx > 0:
                self.rect.right = h.rect.left
            elif self.vx < 0:
                self.rect.left = h.rect.right

        # Вертикаль
        self.apply_gravity()
        prev_bottom = self.rect.bottom
        self.rect.y += int(self.vy)
        self.on_ground = False
        hits = pygame.sprite.spritecollide(self, self.platforms, False)
        for h in hits:
            if self.vy > 0:
                self.rect.bottom = h.rect.top
                self.vy = 0
                self.on_ground = True
                self.jumps_used = 0  # приземлились — сброс количества прыжков
            elif self.vy < 0:
                self.rect.top = h.rect.bottom
                self.vy = 0

        # Опасности
        if pygame.sprite.spritecollideany(self, self.hazards):
            self.alive = False
            return

        # Враги
        enemy_hits = pygame.sprite.spritecollide(self, self.enemies, False)
        for e in enemy_hits:
            falling = self.vy > 0 or self.rect.bottom > prev_bottom
            top_contact = self.rect.bottom - e.rect.top <= max(10, int(self.vy) + 1)
            if falling and top_contact:
                e.kill()
                self.rect.bottom = e.rect.top
                self.vy = -STOMP_BOUNCE
                self.on_ground = False
                # После "стомпа" считаем, что выполнен первый прыжок — доступен ещё один
                self.jumps_used = 1
            else:
                self.alive = False
                return

        # Победа
        if pygame.sprite.spritecollideany(self, self.goals):
            self.won = True

    def update(self, keys):
        if not self.alive or self.won:
            return
        self.handle_input(keys)
        self.move_and_collide()
        if self.rect.top > WORLD_H:
            self.alive = False

# --- Построение уровня ---
def build_level():
    platforms = pygame.sprite.Group()
    hazards = pygame.sprite.Group()
    enemies = pygame.sprite.Group()
    goals = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group()

    player = None

    # Первый проход: твёрдые тайлы и ловушки/флаг
    for row, line in enumerate(LEVEL_MAP):
        for col, ch in enumerate(line):
            x = col * TILE
            y = row * TILE
            if ch == "X":
                p = Platform(x, y)
                platforms.add(p)
                all_sprites.add(p)
            elif ch == "^":
                hz = Hazard(x, y, "spikes")
                hazards.add(hz)
                all_sprites.add(hz)
            elif ch == "~":
                hz = Hazard(x, y, "fire")
                hazards.add(hz)
                all_sprites.add(hz)
            elif ch == "G":
                g = Goal(x, y)
                goals.add(g)
                all_sprites.add(g)

    # Второй проход: игрок и враги
    for row, line in enumerate(LEVEL_MAP):
        for col, ch in enumerate(line):
            x = col * TILE
            y = row * TILE
            if ch == "P":
                player = Player(x, y, platforms, enemies, hazards, goals)
                all_sprites.add(player)
            elif ch == "E":
                e = Enemy(x, y, platforms)
                enemies.add(e)
                all_sprites.add(e)

    return all_sprites, platforms, hazards, enemies, goals, player

# --- Рендер с камерой ---
def draw_with_camera(screen, groups_in_order, cam_x):
    for grp in groups_in_order:
        for spr in grp:
            screen.blit(spr.image, (spr.rect.x - cam_x, spr.rect.y))

def game_loop():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Платформер — двойной прыжок и разнесённые препятствия")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)

    def reset():
        return build_level()

    all_sprites, platforms, hazards, enemies, goals, player = reset()
    game_over = False
    game_won = False

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r and (game_over or game_won):
                    all_sprites, platforms, hazards, enemies, goals, player = reset()
                    game_over = False
                    game_won = False

        keys = pygame.key.get_pressed()

        # Логика
        if not (game_over or game_won):
            all_sprites.update(keys)
            if not player.alive:
                game_over = True
            if player.won:
                game_won = True

        # Камера: центрируем игрока по X, ограничиваем миром
        cam_x = max(0, min(player.rect.centerx - WIDTH // 2, WORLD_W - WIDTH))

        # Рендер
        screen.fill(BG_COLOR)
        draw_with_camera(screen, [platforms, hazards, goals, enemies, pygame.sprite.Group(player)], cam_x)

        hint = "A/D или ←/→ — ходьба, Space — прыжок (×2), R — рестарт, Esc — выход"
        img = font.render(hint, True, TEXT_COLOR)
        screen.blit(img, (16, 8))

        if game_over:
            msg = "Вы пали! Нажмите R для рестарта"
            img2 = font.render(msg, True, (255, 180, 180))
            screen.blit(img2, (16, 40))
        elif game_won:
            msg = "Победа! Достигнут флаг справа. Нажмите R для рестарта"
            img2 = font.render(msg, True, (180, 255, 180))
            screen.blit(img2, (16, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    game_loop()
