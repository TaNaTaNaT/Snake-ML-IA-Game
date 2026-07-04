import os
import argparse
import numpy as np
import torch
from env.snake_env import SnakeEnv
from agents.astar_agent import expert_agent
from agents.model_agent import PolicyNet, collect_expert_data, train_imitation, model_agent_factory
from utils.visual import Visual
import csv
import pygame

# =========================
# Pantalla de espera antes de iniciar el juego
# =========================
def wait_for_start(screen, font):
    """
    Muestra un mensaje 'Press SPACE to start' y espera que el usuario presione SPACE.
    
    Args:
        screen (pygame.Surface): superficie principal de pygame.
        font (pygame.font.Font): fuente para dibujar el texto.
    """
    waiting = True
    screen.fill((0,0,0))
    text = font.render("Press SPACE to start", True, (255,255,255))
    screen.blit(text, (50, 150))
    pygame.display.flip()
    
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

# =========================
# Función principal
# =========================
def main():
    # -------------------------
    # Parseo de argumentos
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true', help='Visualización con pygame')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--collect_episodes', type=int, default=150, help='Episodios para recolectar con A*')
    parser.add_argument('--train_epochs', type=int, default=8, help='Épocas de entrenamiento')
    parser.add_argument('--eval_runs', type=int, default=5, help='Número de evaluaciones del modelo')
    args = parser.parse_args()

    # -------------------------
    # Crear carpetas necesarias
    # -------------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    model_path = "models/snake_model.pt"
    metrics_file = os.path.join("logs","metrics.csv")

    # -------------------------
    # Crear entorno y modelo
    # -------------------------
    env = SnakeEnv(size=10, max_apples=35, seed=args.seed)
    model = PolicyNet(board_size=env.size)

    # -------------------------
    # Cargar modelo existente o entrenar desde el experto
    # -------------------------
    if os.path.isfile(model_path):
        print(f"Cargando modelo entrenado desde {model_path} ...")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("Modelo cargado.")
    else:
        print("Recolectando datos del experto (A*)...")
        dataset = collect_expert_data(env, max_episodes=args.collect_episodes)
        print(f"Datos recolectados: {len(dataset)} pares (obs, action)")

        if len(dataset) > 0:
            print("Entrenando modelo con Imitation Learning (del experto A*) ...")
            train_imitation(model, dataset, epochs=args.train_epochs)
            torch.save(model.state_dict(), model_path)
            print(f"Modelo entrenado y guardado en {model_path}")
        else:
            print("No se recolectaron datos, el modelo no se entrenará.")

    # -------------------------
    # Inicializar visualización
    # -------------------------
    if args.vis:
        vis = Visual(cell=40, size=env.size, fps=10)
        wait_for_start(vis.screen, pygame.font.SysFont('Arial', 30))
    else:
        vis = None

    # -------------------------
    # Preparar CSV de métricas
    # -------------------------
    if not os.path.isfile(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episodio','Agente','Manzanas','Pasos','Tiempo','Motivo'])

    # =========================
    # Evaluación del experto A*
    # =========================
    print("\nEVALUACIÓN: experto (A*)")
    if args.vis:
        env.agent_name = "A*"
    apples, steps, elapsed, reason = env.run_episode(expert_agent, render=not args.vis, vis=vis)
    if args.vis:
        env.elapsed = elapsed
    print(f"Experto terminó: apples={apples}, steps={steps}, tiempo={elapsed:.4f}s, reason={reason}")

    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([1, 'A*', apples, steps, elapsed, reason])

    # =========================
    # Evaluación del modelo (Imitation Learning)
    # =========================
    print("\nEVALUACIÓN: modelo (PyTorch)")
    model_agent = model_agent_factory(model)
    if args.vis:
        env.agent_name = "Modelo"
    apples_m, steps_m, elapsed_m, reason_m = env.run_episode(model_agent, render=not args.vis, vis=vis)
    if args.vis:
        env.elapsed = elapsed_m
    print(f"Modelo terminó: apples={apples_m}, steps={steps_m}, tiempo={elapsed_m:.4f}s, reason={reason_m}")

    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([1, 'Modelo', apples_m, steps_m, elapsed_m, reason_m])

    # =========================
    # Evaluaciones adicionales (timing)
    # =========================
    print("\nEvaluaciones adicionales del modelo:")
    times = []
    for i in range(args.eval_runs):
        envt = SnakeEnv(size=10, max_apples=35, seed=(args.seed or 0) + 100 + i)
        if args.vis:
            envt.agent_name = "Modelo"
        a, s, t, r = envt.run_episode(model_agent, render=False, vis=None)
        print(f" run {i+1}: apples={a}, steps={s}, tiempo={t:.4f}s, reason={r}")
        times.append(t)
    print(f"Tiempo promedio por episodio del modelo: {np.mean(times):.4f}s")

# =========================
# Punto de entrada
# =========================
if __name__ == '__main__':
    main()

