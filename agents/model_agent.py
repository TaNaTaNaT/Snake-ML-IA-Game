import torch
import numpy as np
from env.snake_env import DIRECTIONS, DIR_TO_IDX
from agents.astar_agent import expert_agent, astar_path, path_to_direction
import torch.nn as nn
import torch.optim as optim
import random

# =========================
# Red neuronal para la política
# =========================
class PolicyNet(nn.Module):
    """
    Red neuronal simple (MLP) que recibe el tablero Snake y devuelve logits para las 4 acciones.
    
    Entrada: (B, 2, board_size, board_size)
        - Canal 0: cuerpo de la serpiente
        - Canal 1: manzana
    Salida: (B, 4) logits para acciones [UP, RIGHT, DOWN, LEFT]
    """
    def __init__(self, board_size=10):
        super().__init__()
        in_ch = 2
        flat = in_ch * board_size * board_size
        self.net = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 acciones posibles
        )

    def forward(self, x):
        # Aplana la entrada por batch y pasa por la red
        b = x.shape[0]
        x = x.view(b, -1)
        return self.net(x)

# =========================
# Función utilitaria
# =========================
def obs_to_tensor(obs):
    """
    Convierte la observación (numpy array) a tensor float32 con batch=1.
    
    Parámetros:
        obs (np.ndarray): observación del entorno
    
    Retorna:
        torch.Tensor: tensor con shape (1, 2, S, S)
    """
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

# =========================
# Recolección de datos del experto
# =========================
def collect_expert_data(env, max_episodes=200, max_steps_per_episode=500):
    """
    Genera un dataset (obs, acción) recolectando trayectorias del agente experto (A*).
    
    Para cada paso:
        - Si A* entrega un camino, se toma la acción inicial de ese camino.
        - Si no hay camino, se elige una acción válida al azar (fallback).
        - Se guarda el par (obs, índice_de_acción) y se avanza el entorno.
    
    Criterios de corte:
        - Termina el episodio si el entorno finaliza (done=True)
        - Detiene la recolección total si superamos 2000 muestras
    
    Retorna:
        list[tuple[np.ndarray, int]]: lista de pares (observación, índice de acción)
    """
    dataset = []
    for ep in range(max_episodes):
        obs = env.reset()
        for step in range(max_steps_per_episode):
            path = astar_path(env)
            if path and len(path) >= 2:
                action = path_to_direction(path)
            else:
                # Acción aleatoria válida si A* no propone nada
                acts = env.available_actions()
                action = random.choice(acts) if acts else None

            if action is None:
                break

            # Guardar par (observación, acción_idx)
            dataset.append((obs.copy(), DIR_TO_IDX[action]))
            # Avanzar el entorno
            obs, r, done, info = env.step(action)
            if done:
                break

        # Limitar el tamaño del dataset
        if len(dataset) > 2000:
            break
    return dataset

# =========================
# Entrenamiento por Imitation Learning
# =========================
def train_imitation(model, dataset, epochs=10, batch_size=64, lr=1e-3):
    """
    Entrena la PolicyNet imitando al experto usando CrossEntropy.

    Parámetros:
        model (PolicyNet): red a entrenar
        dataset (list): lista de (obs, action_idx) recolectados del experto
        epochs (int): número de épocas de entrenamiento
        batch_size (int): tamaño de batch
        lr (float): learning rate para Adam
    """
    if not dataset:
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Convertir dataset a arrays
    X = np.array([d[0] for d in dataset])
    Y = np.array([d[1] for d in dataset])
    n = len(X)

    for epoch in range(epochs):
        perm = np.random.permutation(n)  # Barajar datos
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.tensor(X[idx], dtype=torch.float32).to(device)
            yb = torch.tensor(Y[idx], dtype=torch.long).to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.to('cpu')  # Devolver a CPU para guardar/cargar fácilmente

# =========================
# Creación de agente a partir del modelo
# =========================
def model_agent_factory(model):
    """
    Crea un agente callable: action = agent(obs, env)
    
    Lógica:
        - Usa PolicyNet para obtener logits (sin gradientes)
        - Filtra acciones inválidas:
            1) Inverso inmediato
            2) Choque con bordes
            3) Choque con cuerpo (permitiendo mover a la cola)
        - Escoge la acción válida con mayor logit
        - Si ninguna acción válida, fallback con env.available_actions()
    """
    def agent(obs, env):
        tensor = obs_to_tensor(obs)
        with torch.no_grad():
            logits = model(tensor).squeeze(0).numpy()  # (4,)

        # Inicialmente, todas las acciones válidas
        valid = {DIR_TO_IDX[d]: True for d in DIRECTIONS}

        # Prohibir el inverso inmediato
        rev = (-env.direction.dx, -env.direction.dy)
        for d in DIRECTIONS:
            if (d.dx, d.dy) == rev:
                valid[DIR_TO_IDX[d]] = False

        # Filtrar colisiones: bordes o cuerpo (permitir la cola)
        for d in DIRECTIONS:
            idx = DIR_TO_IDX[d]
            if not valid[idx]:
                continue
            hx, hy = env.snake[0]
            nx, ny = hx + d.dx, hy + d.dy
            if not (0 <= nx < env.size and 0 <= ny < env.size):
                valid[idx] = False
            elif (nx, ny) in set(env.snake) and (nx, ny) != env.snake[-1]:
                valid[idx] = False

        # Escoger acción válida con mayor logit
        best_idx = None
        best_val = -1e9
        for idx, ok in valid.items():
            if not ok:
                continue
            if logits[idx] > best_val:
                best_val = logits[idx]
                best_idx = idx

        # Fallback si ninguna acción válida
        if best_idx is None:
            acts = env.available_actions()
            return random.choice(acts) if acts else None

        # Mapear índice -> Direction
        for d in DIRECTIONS:
            if DIR_TO_IDX[d] == best_idx:
                return d

        return None

    return agent
