import os
import cv2
import numpy as np
import heapq
from typing import Tuple, List

Node = Tuple[int, int]

def distance(n1: Node, n2: Node):
    y1, x1 = n1
    y2, x2 = n2
    dx, dy = abs(x1-x2), abs(y1-y2)
    D1, D2 = 10, 14
    return D1*(dx+dy) + (D2 - 2*D1)*min(dx,dy)

def neighbor_nodes(node: Node, grid):
    y, x = node
    nbrs = []
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            ny, nx = y+dy, x+dx
            if (dy or dx) and 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny, nx]:
                nbrs.append((ny, nx))
    return nbrs

def a_star(grid: np.ndarray, start: Node, goal: Node):
    H, W = grid.shape
    h = np.full((H,W), np.inf, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            if grid[i,j]:
                h[i,j] = distance((i,j), goal)
    g = np.full((H,W), np.inf, dtype=np.float64)
    g[start] = 0
    came_from = {}
    open_set = [(h[start], 0, start)]
    while open_set:
        _, g_curr, curr = heapq.heappop(open_set)
        if curr == goal:
            path = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            return path[::-1]
        for nbr in neighbor_nodes(curr, grid):
            cost = 14 if abs(nbr[0]-curr[0]) and abs(nbr[1]-curr[1]) else 10
            tg = g_curr + cost
            if tg < g[nbr]:
                g[nbr] = tg
                f = tg + h[nbr]
                heapq.heappush(open_set, (f, tg, nbr))
                came_from[nbr] = curr
    return []
## scailing func.
Node = Tuple[int, int]

def scale_path(
    path: List[Node],
    orig_size: Tuple[int, int],
    scaled_size: Tuple[int, int]) -> List[Node]:
    
    H, W = orig_size
    scaled_H, scaled_W = scaled_size
    scaled = []
    for y, x in path:
        y2 = int(y * scaled_H / H)
        x2 = int(x * scaled_W / W)
        # 범위 밖 좌표를 0 ~ scaled_H-1/sacled_W-1 사이로 클램프
        y2 = min(scaled_H - 1, max(0, y2))
        x2 = min(scaled_W - 1, max(0, x2))
        scaled.append((y2, x2))
    return scaled

if __name__ == '__main__':
    input_dir = './mazes'
    path_dir  = './paths'
    os.makedirs(path_dir, exist_ok=True)
    
    # just 3:4 ㅋㅋ
    scaled_H = 300
    scaled_W = 400

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        ## naming 1 
        base, _ = os.path.splitext(fname)
        path_file = os.path.join(path_dir, f'{base}.txt')

        if os.path.exists(path_file):
            print(f"Skipping {fname}, path already exists.")
            continue
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        grid = binary.astype(np.uint8)

        H, W = grid.shape
        start = (H-1, W-1)
        goal  = (0,   W-1)
        ## astar code practice
        path = a_star(grid, start, goal)
        # print(path) list => clear
        
        new_path = scale_path(path, (H, W), (scaled_H, scaled_W))
        # print(new_path) tuple => clear
        
        # 480×640 grid and ‘■’ view
        display = [[' ' for _ in range(scaled_W)] for __ in range(scaled_H)]
        
        for (y2, x2) in new_path:
            display[y2][x2] = '■'
            
        sy2 = int(start[0] * scaled_H / H)
        sx2 = int(start[1] * scaled_W / W)
        gy2 = int(goal[0]  * scaled_H / H)
        gx2 = int(goal[1]  * scaled_W / W)   
        display[sy2][sx2] = 'S'
        display[gy2][gx2] = 'G'
        
        # print(display) clear
        with open(path_file, 'w') as f:
            f.write("new path:")
            for a in display:
                for b in a:
                    f.write(f"{b}")
        print(f"Saved path for {fname} → {path_file}")
        # (5.30) fail
