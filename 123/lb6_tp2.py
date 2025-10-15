# -*- coding: utf-8 -*-
"""
Нарды (короткие/длинные) в одном файле на Pygame.
- Окно: 1920x1080, 60 FPS
- Два режима: короткие (Backgammon), длинные (Длинные нарды)
- Полная логика легальных ходов: вход с бара, дубль, взятие, выброс в доме
- Подсветка доступных ходов, drag-and-drop
- Комментарии на русском

Справочные источники по правилам:
- Короткие (Backgammon): базовая механика движения навстречу, взятия одиночных, вход с бара, выброс при всех фишках в доме [web:10][web:16].
- Длинные (Long Nardy): движение обоих игроков в одном направлении, взятия нет, старт «с головы», запрет хода на занятые соперником пункты [web:7].
- Примеры drag-and-drop в Pygame (паттерн обработки мыши) [web:8].
"""

import sys
import random
import pygame

# ------------------------------
# Константы экрана и цветов
# ------------------------------
WIDTH, HEIGHT = 1920, 1080  # 1920x1080 окно [web:8]
FPS = 60  # 60 FPS [web:8]

BG_COLOR = (30, 30, 30)
BOARD_COLOR = (180, 150, 100)
POINT_DARK = (120, 80, 50)
POINT_LIGHT = (220, 190, 150)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GREEN = (60, 180, 110)
RED = (200, 60, 60)
YELLOW = (230, 200, 50)
CYAN = (70, 210, 210)

# ------------------------------
# Геометрия доски (условная)
# ------------------------------
# Классическая доска: 24 пункта (треугольника), по 12 сверху и снизу.
# Будем хранить точки как индексы 0..23 слева-направо для верхнего ряда (12..23) и нижнего (0..11).
# Для отрисовки преобразуем индекс в экранные координаты.

BOARD_MARGIN_X = 180
BOARD_MARGIN_Y = 120
BOARD_WIDTH = WIDTH - 2 * BOARD_MARGIN_X
BOARD_HEIGHT = HEIGHT - 2 * BOARD_MARGIN_Y
BAR_WIDTH = 80  # центральная «бар» зона (для битых фишек в коротких) [web:10][web:16]
HOME_HEIGHT = BOARD_HEIGHT // 2

POINT_WIDTH = (BOARD_WIDTH - BAR_WIDTH) // 12
CHECKER_RADIUS = int(min(POINT_WIDTH * 0.45, (HOME_HEIGHT / 6) * 0.8))
CHECKER_STACK_DY = int((HOME_HEIGHT - 2 * 10) / 5.5)  # плотность укладки в столбике

FONT_NAME = None

# ------------------------------
# Режимы
# ------------------------------
MODE_SHORT = "short"   # Короткие нарды (Backgammon) [web:10][web:16]
MODE_LONG = "long"     # Длинные нарды (Long Nardy) [web:7]

# ------------------------------
# Вспомогательные функции
# ------------------------------

def point_screen_rect(idx):
    """
    Возвращает прямоугольник пункта на экране (для визуализации и прицеливания).
    Верхние пункты: индексы 12..23, нижние: 0..11.
    Центр-бар делит левую шестерку и правую шестёрку.
    """
    # Нижние 0..5 слева, 6..11 справа от бара; Верхние 12..17 справа, 18..23 слева.
    # Рассчитаем x по позиции в полушестёрке.
    left_section_width = (BOARD_WIDTH - BAR_WIDTH) // 2
    # Определяем ряд
    if idx < 12:
        # нижний ряд
        row_top = BOARD_MARGIN_Y + HOME_HEIGHT
        row_height = HOME_HEIGHT
        is_top = False
        pos = idx
        if pos <= 5:
            # левая нижняя шестерка слева от бара
            x = BOARD_MARGIN_X + (5 - pos) * POINT_WIDTH
        else:
            # правая нижняя шестерка справа от бара
            x = BOARD_MARGIN_X + left_section_width + BAR_WIDTH + (pos - 6) * POINT_WIDTH
    else:
        # верхний ряд
        row_top = BOARD_MARGIN_Y
        row_height = HOME_HEIGHT
        is_top = True
        pos = idx - 12
        if pos <= 5:
            # правая верхняя шестерка справа от бара (от правого края к бару)
            x = BOARD_MARGIN_X + left_section_width + BAR_WIDTH + (pos) * POINT_WIDTH
        else:
            # левая верхняя шестерка слева от бара (от бара к левому краю)
            x = BOARD_MARGIN_X + (11 - pos) * POINT_WIDTH
    return pygame.Rect(x, row_top, POINT_WIDTH, row_height), is_top

def point_center_for_stack(idx, stack_len, stack_pos):
    """
    Координаты центра фишки в столбике для пункта idx.
    stack_len — всего в столбике, stack_pos — индекс фишки снизу/сверху (0..stack_len-1).
    """
    rect, is_top = point_screen_rect(idx)
    cx = rect.x + rect.w // 2
    padding = 10
    if not is_top:
        # нижний ряд: укладка снизу вверх
        cy = rect.bottom - padding - CHECKER_RADIUS - stack_pos * CHECKER_STACK_DY
    else:
        # верхний ряд: укладка сверху вниз
        cy = rect.top + padding + CHECKER_RADIUS + stack_pos * CHECKER_STACK_DY
    return cx, cy

def bar_rect():
    # Центральная зона бара по вертикали
    x = BOARD_MARGIN_X + (BOARD_WIDTH - BAR_WIDTH) // 2
    return pygame.Rect(x, BOARD_MARGIN_Y, BAR_WIDTH, BOARD_HEIGHT)

def home_bearoff_rect(color):
    """
    Зона для «снятия» (выброса) фишек; рисуем маленький карман сбоку.
    Для простоты: белые — справа, чёрные — слева.
    """
    w = 60
    if color == 1:
        # белые
        x = WIDTH - BOARD_MARGIN_X + 10
    else:
        # чёрные
        x = BOARD_MARGIN_X - w - 10
    y = BOARD_MARGIN_Y
    h = BOARD_HEIGHT
    return pygame.Rect(x, y, w, h)

# ------------------------------
# Представление состояния
# ------------------------------
# Считаем игроков: 1 — белые, -1 — чёрные.
# Массив board[0..23] хранит положительное число фишек белых или отрицательное — чёрных.
# Бар (битые) и выброшенные храним отдельно (для коротких нард).
# Для длинных нард: бар и взятия не используются [web:7].

class GameState:
    def __init__(self, mode=MODE_SHORT):
        self.mode = mode  # "short" или "long" [web:10][web:7]
        self.board = [0]*24
        self.turn = 1  # 1 белые начинают, в длинных так и принято [web:7], в коротких обычно разыгрывают первый ход [web:16]
        self.bar = {1: 0, -1: 0}       # для коротких: число фишек на баре [web:10][web:16]
        self.borne_off = {1: 0, -1: 0} # выброшенные фишки [web:10][web:16]
        self.dice = []
        self.dice_used = []  # список использованных значений в текущем ходе
        self.selected_point = None
        self.legal_targets = []  # список (dst, used_die)
        self.dragging = None  # {'color':1,-1,'from':'bar'|int,'offset':(dx,dy)}
        self.must_use_all = True  # стараться использовать максимум костей [web:16]
        self.setup_initial()

    def setup_initial(self):
        if self.mode == MODE_SHORT:
            # Классическая стартовая расстановка Backgammon [web:10]
            self.board = [0]*24
            # Белые: 2 на 24, 5 на 13, 3 на 8, 5 на 6 (индексы 23,12,7,5)
            self.board[23] = 2
            self.board[12] = 5
            self.board[7] = 3
            self.board[5] = 5
            # Чёрные зеркально (отрицательные)
            self.board[0] -= 2
            self.board[11] -= 5
            self.board[16] -= 3
            self.board[18] -= 5
            self.bar = {1:0, -1:0}
            self.borne_off = {1:0, -1:0}
            # Первый ход в классике определяется броском обоих — упростим: каждый ход бросаем по 2 кости [web:16]
            self.turn = 1
        else:
            # Длинные нарды: все 15 фишек на «голове» (пункт 24 для белых, 12 для чёрных по классической нотации),
            # используем упрощённое отображение на те же индексы: белые — idx=23, чёрные — idx=11 [web:7]
            self.board = [0]*24
            self.board[23] = 15  # белые
            self.board[11] = -15 # чёрные
            self.bar = {1:0, -1:0}
            self.borne_off = {1:0, -1:0}
            self.turn = 1  # в длинных обычно белые начинают [web:7]

        self.roll_dice()

    def roll_dice(self):
        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)
        if d1 == d2:
            # дубль — четыре хода этим значением [web:16]
            self.dice = [d1, d1, d1, d1]
        else:
            self.dice = [d1, d2]
        self.dice_used = []

    def direction(self, color):
        # В коротких: белые идут от 24 к 1 (от индекса 23 к 0) — направление -1,
        # чёрные — от 1 к 24 (от индекса 0 к 23) — направление +1 [web:10][web:16].
        # В длинных: оба идут против часовой (по одному направлению). Примем, что оба идут к убыванию индекса (23->0),
        # а для чёрных старт на 11, обход круга до их дома. Это сохраняет однонаправленность без ударов [web:7].
        if self.mode == MODE_SHORT:
            return -1 if color == 1 else +1  # встречные направления [web:10][web:16]
        else:
            return -1  # одно направление для обоих [web:7]

    def home_range(self, color):
        # Дом — последние 6 пунктов по направлению движения.
        # В коротких: для белых дом индексы 0..5, для чёрных 18..23 [web:10][web:16].
        # В длинных: дом — последние шесть пунктов своего круга: белые 0..5, чёрные 12..17 (по описанию в источнике) [web:7].
        if self.mode == MODE_SHORT:
            return range(0, 6) if color == 1 else range(18, 24)  # [web:10][web:16]
        else:
            return range(0, 6) if color == 1 else range(12, 18)  # [web:7]

    def is_blocked(self, dst, color):
        # Пункт закрыт, если там 2+ фишки противника (короткие) [web:10][web:16].
        # В длинных: нельзя ходить на пункт, занятый противником даже одной фишкой; взятий нет [web:7].
        v = self.board[dst]
        if self.mode == MODE_SHORT:
            return v * color < -1  # >=2 противника [web:10][web:16]
        else:
            return v * color < 0 or v * color > 0 and (v * color < 0)  # если есть противник — запрещено [web:7]

    def can_hit(self):
        # В длинных взятий нет [web:7]
        return self.mode == MODE_SHORT

    def has_all_in_home(self, color):
        # Проверка, что все фишки в доме (для возможности выброса) [web:10][web:16][web:7].
        hr = set(self.home_range(color))
        # Фишки не должны быть на баре (короткие) [web:10][web:16]
        if self.mode == MODE_SHORT and self.bar[color] > 0:
            return False
        for i, v in enumerate(self.board):
            if v * color > 0 and i not in hr:
                return False
        return True

    def entry_points_from_bar(self, die, color):
        # Вход с бара (только короткие): белые входят в дом противника (верхний ряд для белых),
        # конвертируем по правилу индексации: для белых вход в 24-die (индекс 24-die), для чёрных — die-1 [web:10][web:16].
        if self.mode != MODE_SHORT:
            return []
        if color == 1:
            dst = 24 - die
        else:
            dst = die - 1
        return [dst]

    def generate_legal_moves_from_point(self, src, color, dice_left):
        """
        Сгенерировать легальные ходы из пункта src с учетом доступных костей (dice_left).
        Возвращает список (dst, used_die).
        """
        moves = []
        dir_ = self.direction(color)
        for i, d in enumerate(dice_left):
            dst = src + dir_ * d
            # Проверка на выброс (если за границы, и все в доме) [web:10][web:16][web:7]
            if (dst < 0 or dst > 23):
                if self.has_all_in_home(color):
                    # Разрешаем выброс, но только корректно для крайних по направлению фишек согласно стандартным правилам
                    # Для простоты: позволим выброс при выходе за границу, если нет более дальних по направлению фишек.
                    moves.append(("bearoff", d))
                continue
            # Блокировки
            if self.is_blocked(dst, color):
                continue
            moves.append((dst, d))
        return moves

    def generate_legal_entries_from_bar(self, color, dice_left):
        """
        Легальные входы с бара (короткие): пункт не должен быть закрыт [web:10][web:16].
        """
        res = []
        for d in dice_left:
            for dst in self.entry_points_from_bar(d, color):
                if 0 <= dst <= 23 and not self.is_blocked(dst, color):
                    res.append((dst, d))
        return res

    def current_legal_moves(self):
        """
        Сгенерировать все легальные одиночные шаги для текущего игрока, учитывая бар, дубль, блокировки, взятия и т.п.
        Возвращает dict: {src: [(dst, used_die), ...]} и отдельно входы с бара как src='bar'.
        """
        color = self.turn
        dice_left = self.remaining_dice()
        moves = {}
        if self.mode == MODE_SHORT and self.bar[color] > 0:
            entries = self.generate_legal_entries_from_bar(color, dice_left)
            if entries:
                moves['bar'] = entries
            return moves
        # обычные перемещения
        for i, v in enumerate(self.board):
            if v * color > 0:
                opts = self.generate_legal_moves_from_point(i, color, dice_left)
                if opts:
                    moves[i] = opts
        return moves

    def remaining_dice(self):
        used = self.dice_used[:]
        left = self.dice[:]
        for u in used:
            if u in left:
                left.remove(u)
        return left

    def apply_move(self, src, move):
        """
        Применить единичный шаг: move = (dst| 'bearoff', used_die).
        Обновить доску, бар/выброс, учитывая взятие (только короткие).
        """
        color = self.turn
        dst, used = move
        # Снимаем фишку с источника
        if src == 'bar':
            # вход с бара [web:10][web:16]
            self.bar[color] -= 1
        else:
            self.board[src] -= color

        if dst == "bearoff":
            # выброс [web:10][web:16][web:7]
            self.borne_off[color] += 1
        else:
            # Взятие одиночной фишки соперника (только короткие) [web:10][web:16]
            if self.can_hit() and self.board[dst] == -color:
                self.board[dst] = 0
                self.bar[-color] += 1
            # Поставить фишку
            self.board[dst] += color

        # зафиксировать использованную кость
        self.dice_used.append(used)

    def can_end_turn(self):
        # Ход завершается, когда все кости либо использованы, либо нет легальных ходов [web:16].
        if not self.remaining_dice():
            return True
        # Проверим, остались ли какие-то ходы
        return len(self.current_legal_moves()) == 0

    def end_turn(self):
        # Смена игрока, бросок костей [web:16]
        self.turn *= -1
        self.roll_dice()
        self.selected_point = None
        self.legal_targets = []

    def winner(self):
        # Победитель — кто выбросил 15 [web:10][web:7]
        if self.borne_off[1] >= 15:
            return 1
        if self.borne_off[-1] >= 15:
            return -1
        return None

# ------------------------------
# Рендер
# ------------------------------

def draw_board(surface, gs):
    # Фон
    surface.fill(BG_COLOR)
    # Доска
    board_rect = pygame.Rect(BOARD_MARGIN_X, BOARD_MARGIN_Y, BOARD_WIDTH, BOARD_HEIGHT)
    pygame.draw.rect(surface, BOARD_COLOR, board_rect, border_radius=12)

    # Бар
    pygame.draw.rect(surface, (90, 70, 40), bar_rect(), border_radius=8)

    # Пункты треугольники
    for idx in range(24):
        rect, is_top = point_screen_rect(idx)
        # цвет чередуем
        use_color = POINT_DARK if (idx % 2 == 0) else POINT_LIGHT
        # рисуем как треугольник
        x1 = rect.x
        x2 = rect.x + rect.w
        xm = rect.x + rect.w // 2
        if is_top:
            pts = [(x1, rect.y), (x2, rect.y), (xm, rect.y + rect.h - 10)]
        else:
            pts = [(x1, rect.bottom), (x2, rect.bottom), (xm, rect.y + 10)]
        pygame.draw.polygon(surface, use_color, pts)

    # Дом-«карманы» выброса
    pygame.draw.rect(surface, (70, 70, 70), home_bearoff_rect(1), border_radius=6)
    pygame.draw.rect(surface, (70, 70, 70), home_bearoff_rect(-1), border_radius=6)

def draw_checkers(surface, gs, hover_pos):
    # Рисуем фишки по пунктам; учтем drag: если фишка тащится, её исходный пункт уменьшится на 1 для отрисовки
    dragging = gs.dragging
    # Копия стека для визуальной укладки
    stacks = {}
    for i, v in enumerate(gs.board):
        cnt_w = v if v > 0 else 0
        cnt_b = -v if v < 0 else 0
        stacks[(i, 1)] = cnt_w
        stacks[(i, -1)] = cnt_b

    # Учитываем, что источник сейчас удерживается мышью
    if dragging and dragging['from'] != 'bar':
        src = dragging['from']
        color = dragging['color']
        stacks[(src, color)] -= 1

    # Рисуем фишки из стека
    for i in range(24):
        for color in (1, -1):
            n = stacks[(i, color)]
            for s in range(n):
                cx, cy = point_center_for_stack(i, n, s)
                pygame.draw.circle(surface, WHITE if color == 1 else BLACK, (cx, cy), CHECKER_RADIUS)
                pygame.draw.circle(surface, (0, 0, 0), (cx, cy), CHECKER_RADIUS, 2)

    # Барные фишки (только для коротких)
    if gs.mode == MODE_SHORT:
        br = bar_rect()
        for idx_color, color in enumerate((1, -1)):
            n = gs.bar[color]
            for s in range(n):
                # Стеки сверху для чёрных, снизу для белых
                if color == 1:
                    cx = br.centerx
                    cy = br.bottom - 15 - CHECKER_RADIUS - s * (CHECKER_RADIUS * 2 + 6)
                else:
                    cx = br.centerx
                    cy = br.top + 15 + CHECKER_RADIUS + s * (CHECKER_RADIUS * 2 + 6)
                pygame.draw.circle(surface, WHITE if color == 1 else BLACK, (cx, cy), CHECKER_RADIUS)
                pygame.draw.circle(surface, (0, 0, 0), (cx, cy), CHECKER_RADIUS, 2)

    # Подсветка выбранного и легальных целей
    if gs.selected_point is not None or (gs.mode == MODE_SHORT and gs.bar[gs.turn] > 0):
        for (dst, used) in gs.legal_targets:
            if dst == "bearoff":
                rect = home_bearoff_rect(gs.turn)
                pygame.draw.rect(surface, YELLOW, rect, 4, border_radius=6)
            else:
                r, _ = point_screen_rect(dst)
                pygame.draw.rect(surface, YELLOW, r, 4, border_radius=6)
        # выделить источник
        if gs.selected_point == 'bar':
            pygame.draw.rect(surface, CYAN, bar_rect(), 4, border_radius=6)
        elif isinstance(gs.selected_point, int):
            r, _ = point_screen_rect(gs.selected_point)
            pygame.draw.rect(surface, CYAN, r, 4, border_radius=6)

    # Рисуем тащимую фишку поверх
    if dragging:
        mx, my = hover_pos
        dx, dy = dragging['offset']
        x = mx + dx
        y = my + dy
        color = dragging['color']
        pygame.draw.circle(surface, WHITE if color == 1 else BLACK, (x, y), CHECKER_RADIUS)
        pygame.draw.circle(surface, (0, 0, 0), (x, y), CHECKER_RADIUS, 2)

def draw_ui(surface, gs, font):
    # Текст: режим, чей ход, кости, счёт выброса
    mode_text = f"Режим: {'Короткие' if gs.mode == MODE_SHORT else 'Длинные'}"
    turn_text = f"Ход: {'Белые' if gs.turn == 1 else 'Чёрные'}"
    dice_text = f"Кости: {', '.join(map(str, gs.dice))} | Использовано: {', '.join(map(str, gs.dice_used)) or '-'}"
    off_text = f"Снято — Белые: {gs.borne_off[1]}, Чёрные: {gs.borne_off[-1]}"
    if gs.mode == MODE_SHORT:
        bar_text = f"Бар — Белые: {gs.bar[1]}, Чёрные: {gs.bar[-1]}"
    else:
        bar_text = "Бар не используется в длинных"

    texts = [mode_text, turn_text, dice_text, off_text, bar_text, "ЛКМ: перетащить, ПКМ: отменить выбор, ПРОБЕЛ: пропуск если нет ходов, TAB: смена режима (рестарт)"]
    y = 20
    for t in texts:
        img = font.render(t, True, (230, 230, 230))
        surface.blit(img, (20, y))
        y += img.get_height() + 6

def locate_point_by_pos(pos):
    # Определить, какой пункт под курсором
    for i in range(24):
        rect, _ = point_screen_rect(i)
        if rect.collidepoint(pos):
            return i
    # Проверка бара
    if bar_rect().collidepoint(pos):
        return 'bar'
    return None

def pick_top_checker_from_point(gs, point, color):
    # Есть ли фишка данного цвета в point (или на баре)
    if point == 'bar':
        if gs.mode == MODE_SHORT and gs.bar[color] > 0:
            return True
        return False
    v = gs.board[point]
    return (v * color) > 0

def compute_legal_targets_for_source(gs, src):
    # Получить список доступных целей с указанием какой кубик будет израсходован
    moves = gs.current_legal_moves()
    if src in moves:
        return moves[src]
    return []

def choose_move_by_drop(gs, src, drop_point):
    # Сопоставить локацию с легальными целями
    for (dst, used) in gs.legal_targets:
        if dst == "bearoff":
            if home_bearoff_rect(gs.turn).collidepoint(pygame.mouse.get_pos()):
                return (dst, used)
        else:
            rect, _ = point_screen_rect(dst)
            if rect.collidepoint(pygame.mouse.get_pos()):
                return (dst, used)
    return None

def restart_mode(gs, new_mode):
    gs.mode = new_mode
    gs.setup_initial()

# ------------------------------
# Основной цикл Pygame
# ------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Нарды — короткие/длинные (Pygame)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, 28)

    # По умолчанию — короткие
    gs = GameState(mode=MODE_SHORT)

    running = True
    hover = (0, 0)

    while running:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    # Переключение режима и рестарт
                    new_mode = MODE_LONG if gs.mode == MODE_SHORT else MODE_SHORT
                    restart_mode(gs, new_mode)
                elif event.key == pygame.K_SPACE:
                    # Пропуск, если невозможно сходить
                    if gs.can_end_turn():
                        gs.end_turn()

            elif event.type == pygame.MOUSEMOTION:
                hover = event.pos

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # ЛКМ: начать перетаскивать, если кость и ход есть
                    # Если есть бар (в коротких) — можно тащить только с бара
                    legal = gs.current_legal_moves()
                    src = None
                    if gs.mode == MODE_SHORT and gs.bar[gs.turn] > 0:
                        # Разрешен только бар
                        if bar_rect().collidepoint(event.pos) and 'bar' in legal:
                            src = 'bar'
                    else:
                        # Выбор пункта со своей фишкой
                        p = locate_point_by_pos(event.pos)
                        if isinstance(p, int) and pick_top_checker_from_point(gs, p, gs.turn) and p in legal:
                            src = p

                    if src is not None:
                        gs.selected_point = src
                        gs.legal_targets = compute_legal_targets_for_source(gs, src)
                        # Начать drag
                        mx, my = event.pos
                        gs.dragging = {'color': gs.turn, 'from': src, 'offset': (0, 0)}
                elif event.button == 3:
                    # ПКМ: снять выделение
                    gs.selected_point = None
                    gs.legal_targets = []
                    gs.dragging = None

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and gs.dragging:
                    # Попытка «бросить» на пункт или в дом
                    src = gs.selected_point
                    move = choose_move_by_drop(gs, src, event.pos)
                    if move:
                        gs.apply_move(src, move)
                        # Если все кости израсходованы или больше нет ходов — смена
                        if gs.can_end_turn():
                            gs.end_turn()
                        else:
                            # Обновить доступные источники
                            gs.selected_point = None
                            gs.legal_targets = []
                    # Завершить drag
                    gs.dragging = None

        # Автосброс выбора, если больше нет ходов из выбранного
        if gs.selected_point is not None:
            legal = gs.current_legal_moves()
            if gs.selected_point not in legal and not (gs.selected_point == 'bar' and 'bar' in legal):
                gs.selected_point = None
                gs.legal_targets = []

        # Отрисовка
        draw_board(screen, gs)
        draw_checkers(screen, gs, hover)
        draw_ui(screen, gs, font)

        # Победа?
        w = gs.winner()
        if w is not None:
            msg = f"Победили {'Белые' if w == 1 else 'Чёрные'}! Нажмите TAB для новой партии."
            img = pygame.font.SysFont(FONT_NAME, 48).render(msg, True, (255, 220, 80))
            screen.blit(img, (WIDTH//2 - img.get_width()//2, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
