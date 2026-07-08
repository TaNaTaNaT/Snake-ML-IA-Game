# 🐍 Snake-IA
**Agente de Búsqueda para Resolver Snake**
 
Este repositorio contiene una implementación del clásico juego **Snake (10x10, máx. 35 manzanas)** junto con dos agentes:

- **Agente experto (A\*)**: utiliza búsqueda informada con heurística Manhattan para encontrar caminos óptimos hacia la manzana.
- **Agente modelo (Imitation Learning)**: una red neuronal entrenada a partir de la experiencia del agente A\* para imitar su comportamiento.
## Objetivo del proyecto
- Implementar un **algoritmo de búsqueda** (informado o no) para resolver Snake.
- Evaluar la capacidad del agente para:
  - Evitar colisiones.
  - Recolectar manzanas eficientemente.
- Medir y reportar el **tiempo de resolución** de cada ejecución.
- Documentar el código y explicar la elección del algoritmo.

## 📂 Estructura del repositorio
snake-ia-main/
│── agents/ # Agentes del juego
│ ├── astar_agent.py # Agente experto con A*
│ └── model_agent.py # Red neuronal y funciones de Imitation Learning
│
│── env/
│ └── snake_env.py # Entorno del juego Snake (tablero, reglas, colisiones)
│
│── logs/
│ └── metrics.csv # Registro de resultados (manzanas, pasos, tiempo, motivo)
│
│── models/
│ └── snake_model.pt # Modelo entrenado (PyTorch)
│
│── utils/
│ ├── visual.py # Visualización con pygame (gráficos, HUD, sonidos)
│ ├── loco_manzana.png # Imagen opcional de la manzana
│ ├── rene_hola.mp3 # Sonido de inicio (opcional)
│ └── rene_miau.mp3 # Sonido de game over (opcional)
│
│── snake_ia.py # Script principal de entrenamiento y evaluación
│── requirements.txt # Dependencias necesarias
----------------------------------------------------------------------
🐍Archivo principal
1) snake_ia.py
Rol: Script principal que coordina todo el proyecto.
Qué hace: Inicializa el entorno Snake (SnakeEnv).
Define y entrena (o carga) el modelo neuronal.
Ejecuta y evalúa al agente experto A* y al agente modelo.
Registra métricas en logs/metrics.csv.
Permite visualización gráfica con pygame o ejecución en modo texto.
Uso: se ejecuta desde la terminal con:
python snake_ia.py (modo texto).
python snake_ia.py --vis (modo gráfico).

Carpeta agents/
Contiene los agentes que juegan Snake (es decir, las estrategias para mover la serpiente).
2) astar_agent.py
Implementa el agente A*.
Incluye: manhattan: función heurística (distancia Manhattan).
astar_path: encuentra la ruta más corta hacia la manzana evitando colisiones.
path_to_direction: traduce una ruta en un movimiento (arriba, abajo, izquierda, derecha).
expert_agent: agente experto que toma decisiones usando A*.
Rol: sirve como "maestro" para entrenar al modelo y como baseline de comparación.

3)model_agent.py
Implementa el agente modelo basado en aprendizaje por imitación (Imitation Learning).
Incluye:
PolicyNet: red neuronal en PyTorch.
collect_expert_data: genera dataset de jugadas del agente A*.
train_imitation: entrena el modelo con esos datos.
model_agent_factory: convierte el modelo en un agente que puede interactuar con SnakeEnv.
Rol: agente "aprendiz" que imita al experto y generaliza jugadas.

🌍 Carpeta env/
Contiene la definición del entorno del juego Snake.

4) snake_env.py
Implementa la clase SnakeEnv.
Funcionalidades:
reset: reinicia el tablero, coloca la serpiente y la primera manzana.
step: ejecuta un movimiento, actualiza el estado y calcula recompensas.
available_actions: lista de movimientos válidos desde el estado actual.
run_episode: corre un episodio completo de Snake con un agente dado.
render_text: muestra el tablero en modo texto.
Rol: define las reglas del juego (colisiones, crecimiento, victoria/derrota).

📊 Carpeta logs/

Contiene archivos de registro de métricas.
5) metrics.csv
CSV con resultados de las ejecuciones:
Número de episodio.
Agente utilizado (A* o Modelo).
Manzanas recolectadas.
Pasos realizados.
Tiempo del episodio.
Motivo de finalización (wall, self, max_steps, etc.).
Rol: permite analizar y comparar el desempeño de los agentes.

🧠 Carpeta models/

Contiene el modelo entrenado en PyTorch.
6) snake_model.pt
Archivo binario que guarda los pesos entrenados de la red neuronal.
Permite cargar el modelo sin necesidad de reentrenarlo en cada ejecución.
Rol: persistencia del agente modelo.

🎨 Carpeta utils/

Contiene herramientas de apoyo (visualización y recursos).
7) visual.py
Maneja la visualización del juego con pygame.
Funcionalidades:
Dibuja tablero, serpiente y manzana.
Muestra HUD con manzanas, agente y tiempo.
Muestra mensajes de victoria o derrota.
Reproduce sonidos opcionales (rene_hola.mp3, rene_miau.mp3).
Usa imagen personalizada para la manzana (loco_manzana.png).
Rol: experiencia gráfica para visualizar y presentar el juego.
Recursos adicionales:
loco_manzana.png: imagen para representar la manzana.
rene_hola.mp3: sonido que se reproduce al iniciar.
rene_miau.mp3: sonido de game over.

📦 Otros archivos
8) requirements.txt
Lista de dependencias necesarias para correr el proyecto:
asgiref==3.9.1
Django==5.2.5
filelock==3.19.1
fsspec==2025.9.0
Jinja2==3.1.6
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.5
numpy==2.3.3
pygame==2.6.1
setuptools==80.9.0
sqlparse==0.5.3
sympy==1.14.0
torch==2.8.0
typing_extensions==4.15.0
tzdata==2025.2
Rol: asegura que cualquier usuario pueda instalar el entorno correcto.
Para instalar todas las dependencias necesarias, es necesario ejecutar el siguiente comando: pip install -r requirements.txt

Métricas y evaluación:

Cada ejecución se registra en logs/metrics.csv con:

Episodio: número de corrida.

Agente: "A*" o "Modelo".

Manzanas: recolectadas antes de terminar.

Pasos: movimientos realizados.

Tiempo: duración en segundos.

Motivo: por qué terminó (ej: wall, self, max_steps, done).

Ejemplo: Episodio,Agente,Manzanas,Pasos,Tiempo,Motivo
1,A*,35,311,31.16,done
1,Modelo,1,1000,100.29,max_steps *Se puede ver en metrics.csv*

----------------------------------------------------------------------
## ⚙️ Instalación
1. Clonar este repositorio o descargar el `.zip`.
2. Abrir el proyecto en **Visual Studio Code**.
3. Crear un entorno virtual (recomendado):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
'''En caso de aparecer error ejecutar en PowerShell como admin: Set-ExecutionPolicy RemoteSigned
4. Instalar dependencias: pip install -r requirements.txt
5. Ejecutar juego: python snake_ia.py --vis
----------------------------------
Durante el juego verás:
Presiona SPACE para comenzar.
🟩 Serpiente (cabeza más intensa, cuerpo degradado).

🍎 Manzana (imagen).

HUD con:
Manzanas comidas.
Nombre del agente (A* o Modelo).
Tiempo del episodio.
-----------------------------

Algoritmos utilizados
1. Agente Experto (A*)
Tipo: Búsqueda informada.
Heurística: Distancia Manhattan (|x1 - x2| + |y1 - y2|).

Ventajas:
Encuentra caminos cortos hacia la manzana.
Evita chocar contra la pared o el cuerpo.

1.- Óptimo en trayectorias → siempre encuentra la ruta más corta desde la cabeza de la serpiente hasta la manzana, evitando rodeos innecesarios.
2.- Evita muertes innecesarias → a diferencia de un agente greedy o aleatorio, considera colisiones con cuerpo y paredes, reduciendo la probabilidad de perder de forma tonta.
3.- Sirve como profesor para el modelo → genera un dataset estable y confiable para entrenar la red neuronal. Si el experto fuese inconsistente, el modelo aprendería mal.
4.- Simplicidad de implementación → balancea eficiencia, calidad de las decisiones y facilidad de programar/debuggear en comparación con métodos más complejos como MCTS o Q-learning.
5.- Cumple con los objetivos del proyecto → el enunciado pedía un algoritmo de búsqueda y A* es el estándar clásico en este tipo de problemas.

Limitaciones: En escenarios donde la serpiente ocupa gran parte del tablero, puede quedarse sin soluciones viables y terminar atrapada.

2. Agente Modelo (Imitation Learning)

Tipo: Red neuronal feed-forward (PyTorch):
Arquitectura: Input: tablero (2 canales: serpiente y manzana).
Capas ocultas: 256 → 128 → 4 salidas (acciones).

Entrenamiento: Dataset generado con el agente A*.
Función de pérdida: CrossEntropy.
Optimizador: Adam.

Ventajas:

Generaliza jugadas sin necesidad de calcular el A* en cada paso.
Más rápido en ejecución.

¿Por qué un modelo de Imitation Learning?
1.- Aprende de un experto confiable → A* actúa como maestro, permitiendo que el modelo generalice buenas jugadas.
2.- Mayor velocidad de ejecución → una vez entrenado, el modelo decide acciones sin necesidad de recalcular una búsqueda A* en cada paso.
3.- Capacidad de generalizar → puede encontrar jugadas razonables en estados no vistos durante el entrenamiento.
4.- Integración con técnicas modernas de IA → conecta la parte de búsqueda clásica con métodos de aprendizaje automático.

Limitaciones: No siempre garantiza la mejor ruta y su rendimiento depende directamente de la calidad y cantidad de los datos de entrenamiento recolectados del Dataset (agente A*).

------------------------------------------------------------------------------
