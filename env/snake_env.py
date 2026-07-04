import numpy as np
import random
from collections import deque, namedtuple
import time

# =========================
# Representación de direcciones
# =========================
Direction = namedtuple('Direction', ['dx', 'dy'])
UP = Direction(0, -1)
DOWN = Direction(0, 1)
LEFT = Direction(-1, 0)
RIGHT = Direction(1, 0)

# Orden canónico de acciones: ↑ → ↓ ←
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
# Mapeo Direction -> índice {0,1,2,3} para redes o arrays
DIR_TO_IDX = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}

# =========================
# Entorno del juego Snake
# =========================
class SnakeEnv:
    """
    Entorno Snake sobre grilla cuadrada.
    
    Tablero: size x size
    Estado: posiciones ocupadas por la serpiente y la manzana
    Movimientos: 4 direcciones (sin diagonales)
    Recompensas:
        +1 al comer manzana
        -1 al chocar con pared o cuerpo
         0 en paso normal
    Terminación:
        done=True por colisión o al alcanzar max_apples (victoria)
    """
    def __init__(self, size=10, max_apples=35, seed=None):
        self.size = size
        self.max_apples = max_apples
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    # =========================
    # Reinicio del entorno
    # =========================
    def reset(self):
        """
        Reinicia el estado a un juego nuevo.
        Coloca serpiente en el centro (longitud 3), dirección LEFT, 
        manzana en celda vacía aleatoria.
        """
        self.snake = deque()
        cx, cy = self.size // 2, self.size // 2
        # Longitud inicial 3, extendiéndose hacia la derecha
        self.snake.appendleft((cx, cy))       # cabeza
        self.snake.append((cx + 1, cy))
        self.snake.append((cx + 2, cy))       # cola
        self.direction = LEFT
        self.apples_eaten = 0
        self.apple = self._place_apple()
        self.done = False
        self.won = False
        self.steps = 0
        return self._get_obs()

    # =========================
    # Colocación de manzana
    # =========================
    def _place_apple(self):
        """
        Coloca la manzana en una celda vacía.
        Retorna: coordenadas (x,y) o None si no hay celdas libres.
        """
        empty = {(x, y) for x in range(self.size) for y in range(self.size)} - set(self.snake)
        if not empty:
            self.apple = None
            self.done = True
            return None
        self.apple = random.choice(list(empty))
        return self.apple

    # =========================
    # Observación del estado
    # =========================
    def _get_obs(self):
        """
        Representa el tablero como un tensor 2D de canales.
        Canal 0: 1.0 donde está la serpiente
        Canal 1: 1.0 donde está la manzana
        """
        grid = np.zeros((2, self.size, self.size), dtype=np.float32)
        for (x, y) in self.snake:
            grid[0, y, x] = 1.0
        if self.apple is not None:
            ax, ay = self.apple
            grid[1, ay, ax] = 1.0
        return grid

    # =========================
    # Acciones válidas
    # =========================
    def available_actions(self):
        """
        Devuelve lista de direcciones válidas desde el estado actual.
        Regla: No permitir reversa inmediata, colisión con pared o cuerpo (salvo la cola)
        """
        acts = []
        for d in DIRECTIONS:
            # No permitir reversa inmediata
            if (d.dx == -self.direction.dx and d.dy == -self.direction.dy):
                continue
            hx, hy = self.snake[0]
            nx, ny = hx + d.dx, hy + d.dy
            # Fuera de límites
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                continue
            # Colisión con cuerpo (permitimos moverse a la cola que se liberará)
            if (nx, ny) in set(self.snake) and (nx, ny) != self.snake[-1]:
                continue
            acts.append(d)
        return acts

    # =========================
    # Paso del entorno
    # =========================
    def step(self, action_direction):
        """
        Aplica una acción y avanza un paso.
        Retorna:
            obs (np.ndarray): observación siguiente
            reward (float): recompensa
            done (bool): True si terminó el episodio
            info (dict): {'reason': 'wall'|'self'}
        """
        if self.done or self.won:
            return self._get_obs(), 0.0, True, {}

        # Evitar reversa inmediata
        if (action_direction.dx == -self.direction.dx and action_direction.dy == -self.direction.dy):
            action_direction = self.direction
        self.direction = action_direction

        hx, hy = self.snake[0]
        nx, ny = hx + self.direction.dx, hy + self.direction.dy

        # Colisión pared
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            self.done = True
            return self._get_obs(), -1.0, True, {'reason':'wall'}
        
        # Colisión cuerpo
        hit_self = (nx, ny) in set(self.snake)
        grows = (self.apple == (nx, ny))
        if hit_self and not (not grows and (nx, ny) == self.snake[-1]):
            self.done = True
            return self._get_obs(), -1.0, True, {'reason':'self'}

        # Movimiento / crecimiento
        self.snake.appendleft((nx, ny))
        reward = 0.0
        if grows:
            self.apples_eaten += 1
            if self.apples_eaten >= self.max_apples:
                self.won = True
                reward = 1.0
            else:
                self._place_apple()
                reward = 1.0
        else:
            self.snake.pop()

        self.steps += 1
        return self._get_obs(), reward, self.done or self.won, {}

    # =========================
    # Bucle de episodio completo
    # =========================
    def run_episode(self, agent_fn, render=False, vis=None, max_steps=1000, sleep=0.05):
        """
        Ejecuta un episodio completo hasta terminar (colisión, victoria o max_steps)
        
        Parámetros:
            agent_fn (callable): función agente(obs, env) -> Direction | None
            render (bool): True imprime tablero en texto
            vis (Visual|None): objeto pygame para UI
            max_steps (int): tope de pasos
            sleep (float): retardo por paso
        
        Retorna:
            tuple: (manzanas_comidas, pasos, tiempo_total, reason)
        """
        obs = self.reset()
        start_time = time.perf_counter()
        steps = 0

        while True:
            action = agent_fn(obs, self)
            if action is None:
                reason = 'no_action'
                break
            obs, r, done, info = self.step(action)
            steps += 1

            # Render texto
            if render and not vis:
                print(self.render_text())
                print('steps', steps, 'apples', self.apples_eaten)
                print('---')

            # Render pygame
            if vis:
                vis.draw(self)

            if done:
                reason = info.get('reason', 'done')
                break
            if steps >= max_steps:
                reason = 'max_steps'
                break
            if sleep > 0:
                time.sleep(sleep)

        elapsed = time.perf_counter() - start_time
        return self.apples_eaten, steps, elapsed, reason

    # =========================
    # Representación ASCII del tablero
    # =========================
    def render_text(self):
        """
        Retorna string con el tablero en modo texto:
            'S' = cabeza, 's' = cuerpo, 'A' = manzana, '.' = vacío
        """
        s = [['.' for _ in range(self.size)] for __ in range(self.size)]
        for (x, y) in self.snake:
            s[y][x] = 's'
        hx, hy = self.snake[0]
        s[hy][hx] = 'S'
        if self.apple is not None:
            ax, ay = self.apple
            s[ay][ax] = 'A'
        return '\n'.join(''.join(row) for row in s)

