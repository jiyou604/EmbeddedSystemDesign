import numpy as np
import heapq
from typing import Tuple

Node = Tuple[int, int]

matrix = [
    [1, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1],
]
start = (0, 0)
target = (6, 6)

map_np = np.array(matrix)
ROWS, COLS = map_np.shape

# 거리 제곱이 아니라 정수 기반 거리 사용
def distance(node1: Node, node2: Node):
    y1, x1 = node1
    y2, x2 = node2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    D = 10
    D2 = 14
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def neighbor_nodes(current_node: Node):
    neighbors = []
    delta = [-1, 0, 1]
    for dx in delta:
        for dy in delta:
            nx, ny = current_node[0] + dx, current_node[1] + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and map_np[nx, ny]:
                neighbors.append((nx, ny))
    return neighbors

def path(map_np: np.ndarray, start: Node, destination: Node):
    col, row = map_np.shape
    heuristic = np.zeros_like(map_np, dtype=np.int32)

    for i in range(col):
        for j in range(row):
            if map_np[i, j]:
                heuristic[i, j] = distance((i, j), destination)

    g_score = np.full_like(map_np, np.inf, dtype=np.float64)
    g_score[start] = 0

    came_from = {}
    open_set = []
    heapq.heappush(open_set, (heuristic[start], 0, start))

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == destination:
            # 경로 재구성
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path = path[::-1]

            # 시각화
            visual = map_np.astype(str)
            for y, x in path:
                visual[y, x] = "*"
            sy, sx = start
            ty, tx = destination
            visual[sy, sx] = "S"
            visual[ty, tx] = "G"
            for row in visual:
                print(" ".join(row))
            return path

        for neighbor in neighbor_nodes(current):
            if neighbor == current:
                continue

            is_diagonal = abs(neighbor[0] - current[0]) == 1 and abs(neighbor[1] - current[1]) == 1
            step_cost = 14 if is_diagonal else 10

            tentative_g = current_g + step_cost
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic[neighbor]
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                came_from[neighbor] = current

    print("경로를 찾을 수 없습니다.")
    return []

path(map_np, start, target)
