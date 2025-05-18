import random as rd
class room: 
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.dir = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        rd.shuffle(self.dir)

    def current(self):
        return self.x, self.y
    def next(self):
        return self.dir.pop()
    
def makemaze():
    ## size = rd.randint(1,5)
    size = 3
    rooms = [[room(x,y) for x in range(size)] for y in range(size)]
    maze = [['■' for f in range(size*2 + 1)] for f in range(size*2 + 1)]

    visited = []

    def make(currentroom):
        cx, cy = currentroom.current()
        visited.append((cx, cy))
        maze[cy*2 + 1][cx*2 + 1] = '   '
        while currentroom.dir:
            nx, ny = currentroom.next()
            if 0 <= nx < size and 0 <= ny < size:
                if (nx, ny) not in visited:
                    maze[cy+ny+1][cx+nx+1]='   '
                    make(rooms[ny][nx])

    make(rooms[0][0])

    return maze

def savemaze(maze, filename):
    with open(filename, 'w') as file:
        for y in range(len(maze)):
            for x in range(len(maze)):
                file.write(maze[y][x])
            file.write('\n')
            

if __name__ == '__main__':
    maze = makemaze()
    savemaze(maze, 'maze.txt')
