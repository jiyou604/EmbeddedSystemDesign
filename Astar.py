import numpy as np
import heapq
import cv2
from typing import Tuple, List, Dict

Node = Tuple[int, int]

class AStarPathfinder:
    def __init__(
        self,
        image: np.ndarray,              # Grayscale image
        start: Node,
        goal: Node,
        threshold: int = 50             # 픽셀 값이 이보다 작으면 장애물
    ):
        assert len(image.shape) == 2, "입력은 grayscale 이미지여야 합니다."
        self.map_np = (image >= threshold).astype(np.uint8)  # 밝으면 1, 어두우면 0

        self.ROWS, self.COLS = self.map_np.shape
        self.start = start
        self.goal = goal

    def heuristic(self, a: Node, b: Node) -> int:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def get_neighbors(self, pos: Node) -> List[Node]:
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4방향
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.ROWS and 0 <= ny < self.COLS and self.map_np[nx, ny] == 1:
                neighbors.append((nx, ny))
        return neighbors

    def get_path(self) -> List[Node]:
        open_set = []
        heapq.heappush(open_set, (self.heuristic(self.start, self.goal), 0, self.start))

        came_from: Dict[Node, Node] = {}
        g_score = {self.start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == self.goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                tentative_g = current_g + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    came_from[neighbor] = current

        return []

    def print_path(self, path: List[Node], scale: int = 300):
        """
        path: 경로 리스트
        scale: 출력 시 확대 배율 (기본 5배)
        """
        import cv2

        # 시각화용 컬러 이미지 (RGB)
        vis = np.stack([self.map_np * 255] * 3, axis=-1).astype(np.uint8)

        for x, y in path:
            vis[x, y] = [0, 255, 0]  # 경로는 초록색
        sx, sy = self.start
        gx, gy = self.goal
        vis[sx, sy] = [0, 0, 255]   # 시작점은 빨간색
        vis[gx, gy] = [255, 0, 0]   # 도착점은 파란색

        # 확대해서 보기 좋게 출력
        vis_scaled = cv2.resize(vis, (scale, scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Path", vis_scaled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import cv2

# 예시용 grayscale 이미지 로드
gray_img = cv2.imread("./mazes/1881481-800175472.png", cv2.IMREAD_GRAYSCALE)

start_pixel = (100, 150)
goal_pixel = (500, 600)

finder = AStarPathfinder(gray_img, start=start_pixel, goal=goal_pixel, threshold=50)

path = finder.get_path()
finder.print_path(path)

