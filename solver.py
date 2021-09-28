# Solver Functions for Sudoku Solver


class Solver:
    def __init__(self, grid):
        self.grid = grid
        self.solve()

    def solve(self, grid=None):

        if grid is None:
            grid = self.grid
        find = self.find_empty(grid)
        if not find:
            return True
        else:
            x, y = find

        for p in range(1, 10):
            if self.possible(grid, y, x, p):
                grid[y][x] = p
                if self.solve(grid):
                    self.grid = grid
                    return True
                grid[y][x] = 0
        return False

    @staticmethod
    def possible(grid, y, x, p):
        for i in range(0, 9):
            if grid[y][i] == p:
                return False
            if grid[i][x] == p:
                return False
        x0 = (x//3)*3
        y0 = (y//3)*3

        for i in range(0, 3):
            for j in range(0, 3):
                if grid[y0+i][x0+j] == p:
                    return False
        return True

    @staticmethod
    def find_empty(grid):
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] == 0:
                    return x, y


