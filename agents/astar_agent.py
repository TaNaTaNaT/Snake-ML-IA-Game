from env.snake_env import DIRECTIONS  # Importa las direcciones posibles (UP, DOWN, LEFT, RIGHT)
from heapq import heappush, heappop   # Importa funciones para manejar una cola de prioridad (heap)

def manhattan(a, b):
    """
    Calcula la distancia Manhattan entre dos puntos en el tablero.

    Parámetros:
        a (tuple): Coordenadas (x, y) del primer punto.
        b (tuple): Coordenadas (x, y) del segundo punto.

    Retorna:
        int: Distancia de Manhattan entre 'a' y 'b'.
    """
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_path(env):
    """
    Implementa el algoritmo A* para encontrar un camino desde la cabeza de la serpiente
    hasta la manzana actual en el tablero.

    Parámetros:
        env (SnakeEnv): Instancia del entorno Snake.

    Retorna:
        list[tuple] | None: Lista de coordenadas que forman el camino hacia la manzana,
                            o None si no existe un camino válido.
    """
    start = env.snake[0]  # Cabeza de la serpiente
    goal = env.apple       # Posición de la manzana
    if goal is None:
        return None       # No hay manzana, no hay camino

    blocked = set(env.snake)  # Posiciones ocupadas por la serpiente
    open_set = []             # Cola de prioridad de nodos abiertos
    # Cada elemento de open_set: (f_score = g+h, g_score, nodo actual, nodo padre)
    heappush(open_set, (manhattan(start, goal), 0, start, None))

    came_from = {}  # Diccionario para reconstruir el camino
    gscore = {start: 0}  # Coste acumulado desde el inicio
    visited = set()      # Nodos ya explorados

    while open_set:
        f, g, current, parent = heappop(open_set)  # Sacar el nodo con menor f_score
        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent  # Guardar de dónde llegamos

        # Si llegamos a la manzana, reconstruir el camino
        if current == goal:
            path = []
            cur = current
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()  # De inicio a fin
            return path

        cx, cy = current
        for d in DIRECTIONS:  # Explorar vecinos: arriba, derecha, abajo, izquierda
            nx, ny = cx + d.dx, cy + d.dy
            neighbor = (nx, ny)

            # Verificar límites del tablero
            if not (0 <= nx < env.size and 0 <= ny < env.size):
                continue

            # Permitir moverse a la cola, porque se moverá en el siguiente paso
            tail = env.snake[-1]
            if neighbor in blocked and neighbor != tail:
                continue

            tentative_g = g + 1  # Costo acumulado hasta el vecino
            # Si el vecino es nuevo o encontramos un camino más corto
            if neighbor not in gscore or tentative_g < gscore[neighbor]:
                gscore[neighbor] = tentative_g
                heappush(open_set, (tentative_g + manhattan(neighbor, goal), tentative_g, neighbor, current))

    return None  # No se encontró un camino válido

def path_to_direction(path):
    """
    Convierte un camino (lista de coordenadas) en la dirección inicial
    que debe tomar la serpiente.

    Parámetros:
        path (list[tuple]): Lista de coordenadas del camino.

    Retorna:
        Direction | None: Dirección del primer movimiento o None si no hay camino.
    """
    if path is None or len(path) < 2:
        return None

    (x0, y0), (x1, y1) = path[0], path[1]
    dx, dy = x1 - x0, y1 - y0

    for d in DIRECTIONS:
        if d.dx == dx and d.dy == dy:
            return d
    return None

def expert_agent(obs, env):
    """
    Agente "experto" que utiliza A* para decidir la siguiente acción.

    Parámetros:
        obs (np.array): Observación del entorno (estado del tablero).
        env (SnakeEnv): Instancia del entorno Snake.

    Retorna:
        Direction | None: Dirección elegida para el próximo movimiento.
    """
    path = astar_path(env)        # Calcular el camino óptimo usando A*
    d = path_to_direction(path)   # Convertir camino en dirección de movimiento

    if d is None:
        # Si A* no encuentra camino, elegir la primera acción válida disponible
        acts = env.available_actions()
        d = acts[0] if acts else None

    return d
