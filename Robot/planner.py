from typing import List, Tuple, Dict
import math
import pyastar2d
import numpy as np
import time
from tilsdk.localization import *
from matplotlib import pyplot as plt


class MyPlanner:
    def __init__(self, map_: SignedDistanceGrid = None, waypoint_sparsity=0.5, optimize_threshold=3, consider=4, biggrid_size=0.8):
        '''
        Parameters
        ----------
        map : SignedDistanceGrid
            Distance grid map
        sdf_weight: float
            Relative weight of distance in cost function.
        waypoint_sparsity: float
            0.5 results in every 50th waypoint being taken at scale=0.01 and 10th at scale=0.05
        consider: float
            For the get_explore function only. See there for more details.
        biggrid_size:
            Divide the grid into squares of side length biggrid_size m.
            When there are no clues, the planner will try to explore every square of this big grid.
        '''
        # ALL grids (including big_grid use [y][x] convention)
        self.optimize_threshold = optimize_threshold
        self.map = map_
        self.bgrid = self.transform_add_border(self.map.grid.copy())  # Grid which takes the borders into account
        self.astar_grid = self.transform_for_astar(self.bgrid.copy())
        self.waypoint_sparsity = waypoint_sparsity
        self.biggrid_size = biggrid_size
        self.bg_idim = math.ceil(5 / biggrid_size)  # i:y
        self.bg_jdim = math.ceil(7 / biggrid_size)  # j:x
        self.big_grid = [[0 for j in range(self.bg_jdim)] for i in range(self.bg_idim)]  # Big_grid stores whether each 0.5*0.5m tile of the arena has been visited
        self.big_grid_centre = [[0 for j in range(self.bg_jdim)] for i in range(self.bg_idim)]
        self.consider = consider
        self.passable = self.map.grid > 0
        self.path = None
        self.plt_init = False
        self.scat = None

        for i in range(self.bg_idim):
            for j in range(self.bg_jdim):
                # Find the closest free location to centre of this cell
                y_pos = min(4.9, i * self.biggrid_size + self.biggrid_size / 2)
                x_pos = min(6.9, j * self.biggrid_size + self.biggrid_size / 2)

                grid_loc = self.map.real_to_grid(RealLocation(x_pos, y_pos))
                grid_loc = grid_loc[1], grid_loc[0]
                nc = self.nearest_clear(grid_loc, self.passable)
                # If the closest free location to the entre of the cell is in another cell,
                # ignore this cell by marking it as visited
                # This doesn't happen though
                nc = nc[1], nc[0]
                nc = self.map.grid_to_real(nc)
                # print("gridctr",RealLocation(x_pos,y_pos),"nc",nc)
                if self.big_grid_of(nc) != (j, i):
                    self.big_grid[i][j] = 100
                else:
                    self.big_grid_centre[i][j] = nc

    def transform_add_border(self, og_grid):
        grid = og_grid.copy()
        a, b = grid.shape
        for i in range(a):
            for j in range(b):
                grid[i][j] = min(grid[i][j], i + 1, a - i, j + 1, b - j)
        return grid

    def transform_for_astar(self, grid):
        # Possible to edit this transform if u want
        k = 500  # tune this for sensitive to stay away from wall. Lower means less sensitive -> allow closer to walls
        grid2 = grid.copy()
        # grid2[grid2 > 0] = 1 + k / (grid2[grid2 > 0])
        # grid2[grid2 <= 0] = np.inf
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid2[i][j] > 0:
                    grid2[i][j] = max(1 + k / grid2[i][j], 1 + k / 14.5)  # set area further than 14.5cm away from the wall to be of equal potential
                else:
                    grid2[i][j] = np.inf
        return grid2.astype("float32")

    def big_grid_of(self, l: RealLocation):  # Returns the big grid array indices of a real location
        return int(l[0] // self.biggrid_size), int(l[1] // self.biggrid_size)

    def visit(self, l: RealLocation):
        indices = self.big_grid_of(l)
        # print("Location:", l)
        # print("indices:", indices)
        # print("bg_idim:", self.bg_idim)
        # print("bg_jdim:", self.bg_jdim)
        indices = (min(indices[0], self.bg_jdim - 1), min(indices[1], self.bg_idim - 1))
        self.big_grid[indices[1]][indices[0]] = max(1, self.big_grid[indices[1]][indices[0]])

    def get_explore(self, l: RealLocation, debug: bool = False):  # Call this to get a location to go to if there are no locations of interest left
        # debug: Whether to plot maps and show info
        # consider (in __init__): Consider the astar paths of this number of the closest unvisited cells by euclidean distance
        # Larger number gives better performance but slower
        m = 100
        for i in range(self.bg_idim):
            for j in range(self.bg_jdim):
                m = min(m, self.big_grid[i][j])
        distance = []
        for i in range(self.bg_idim):
            for j in range(self.bg_jdim):
                if self.big_grid[i][j] == m:
                    distance.append((self.heuristic(self.big_grid_centre[i][j], l), (i, j)))
        distance.sort()

        if len(distance) == 0:
            return None

        distance = distance[:min(self.consider, len(distance))]
        for i in range(len(distance)):
            loc = self.big_grid_centre[distance[i][1][0]][distance[i][1][1]]
            if debug:
                print("l, loc:", l, loc)
            path = self.plan(l, loc, whole_path=True, display=debug)
            distance[i] = (1e18 if path is None else len(path), distance[i][1])
            if debug:
                print("Path length:", distance[i][0])

        distance.sort()
        if debug:
            print("Closest guys", distance[:min(5, len(distance))])

        closest = distance[0]
        self.big_grid[closest[1][0]][closest[1][1]] += 1

        if debug:
            plt.imshow(self.big_grid)
            plt.title("Big grid now")
            plt.show()
        return self.big_grid_centre[closest[1][0]][closest[1][1]]

    def heuristic(self, a: GridLocation, b: GridLocation) -> float:
        '''Planning heuristic function.
        Parameters
        ----------
        a: GridLocation
            Starting location.
        b: GridLocation
            Goal location.
        '''
        return euclidean_distance(a, b)

    def nearest_clear(self, loc, passable):
        '''Utility function to find the nearest clear cell to a blocked cell'''
        loc = min(loc[0],self.map.grid.shape[0]-1),min(loc[1],self.map.grid.shape[1]-1)
        loc = max(loc[0],0),max(loc[1],0)
        if not passable[loc]:
            best = (1e18, (-1, -1))
            for i in range(self.map.height):  # y
                for j in range(self.map.width):  # x
                    if self.map.grid[(i, j)] > 0:
                        best = min(best, (self.heuristic(GridLocation(i, j), loc), (i, j)))
            loc = best[1]
        return loc

    def plan(self, start: RealLocation, goal: RealLocation, whole_path: bool = False, display: bool = False) -> List[RealLocation]:
        '''Plan in real coordinates.

        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: RealLocation
            Starting location.
        goal: RealLocation
            Goal location.
        whole_path:
            Whether to return the whole path instead of version with select waypoints
        display:
            Whether to visualise the path

        Returns
        -------
        path
            List of RealLocation from start to goal.
        '''
        self.path = self.plan_grid(self.map.real_to_grid(start), self.map.real_to_grid(goal), whole_path)
        if self.path is None:
            return None
        self.path = self.path + self.path[-1:] # Duplicate destination wp to avoid bug in main loop which happens for the final waypoint as the path list is empty
        if display:
            gridpath = [self.map.real_to_grid(x) if isinstance(x, RealLocation) else x for x in self.path]
            self.gridpathx = [x[0] for x in gridpath]
            self.gridpathy = [x[1] for x in gridpath]
            self.visualise_path()
        self.path = [self.map.grid_to_real(wp) for wp in self.path]
        return self.path

    def plan_grid(self, start: GridLocation, goal: GridLocation, whole_path: bool = False, debug: bool = False) -> List[GridLocation]:
        '''Plan in grid coordinates.

        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: GridLocation
            Starting location.
        goal: GridLocation
            Goal location.
        whole_path:
            Whether to return the whole path instead of version with select waypoints
        debug:
            Whether to print start and end locations
        Returns
        -------
        path
            List of GridLocation from start to goal.
        '''

        if not self.map:
            raise RuntimeError('Planner map is not initialized.')

        start = start[1], start[0]
        goal = goal[1], goal[0]  # Use i=x,j=y convention for convenience
        passable = self.map.grid > 0

        if debug:
            print("original start", start)
            print("original goal", goal)
        start = self.nearest_clear(start, passable)
        goal = self.nearest_clear(goal, passable)
        if debug:
            print("start", start)
            print("goal", goal)

        # astar
        path = pyastar2d.astar_path(self.astar_grid, start, goal, allow_diagonal=True)
        if path is None:
            return None
        coeff = int(self.waypoint_sparsity / self.map.scale)  # default sparsity 0.5 --> 50 for 0.01, 10 for 0.05
        path = list(path)
        path = [(x[1], x[0]) for x in path]
        if whole_path:
            return path
        coeff = max(coeff, 1)
        path = path[:1] + path[:-1:coeff] + path[-1:]  # Take the 1st, last, and every 20th waypoint in the middle
        # Duplication of last waypoint to avoid bug in main loop has been moved to plan()
        return self.optimize_path(path)

    def optimize_path(self, path: List[GridLocation]) -> List[GridLocation]:
        new_path = [path[0]]  # starting point always in path
        for i in range(1, len(path) - 1, 1):
            if not ((abs(path[i - 1][0] - path[i][0]) < self.optimize_threshold and abs(path[i][0] - path[i + 1][0]) < self.optimize_threshold) or
                    (abs(path[i - 1][1] - path[i][1]) < self.optimize_threshold and abs(path[i][1] - path[i + 1][1]) < self.optimize_threshold)):  # 3 consecutive points are on a straight line in either x or y direction
                new_path.append(path[i])
        new_path.append(path[-1])  # add last point
        return new_path

    def wall_within_1m(self, l: RealLocation, angle: int) -> bool:
        # Angle is expected to be 0, 90, 180, or 270
        direction = int(round(angle / 90, 2))
        g = self.map.real_to_grid(l)
        x, y = g[0], g[1]
        its = math.ceil(1 / self.map.scale)
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        for it in range(its):
            x += dx[direction]
            y += dy[direction]
            if x < 0 or y < 0 or x >= self.map.grid.shape[1] or y >= self.map.grid.shape[0]:
                return True  # We at border
            if self.map.grid[y][x] <= 0:
                return True
        return False

    def visualise_path(self):
        if not self.plt_init:
            pathmap = self.map.grid.copy()
            pathmap = pathmap > 0
            pathmap = pathmap.astype("int32")
            plt.imshow(pathmap)
            plt.title("Path: white -> start, black -> end")
            self.scat = plt.scatter(self.gridpathx, self.gridpathy, c=np.arange(len(self.gridpathx)), cmap='Greys')
            self.plt_init = True
        else:
            self.scat.remove()
            self.scat = plt.scatter(self.gridpathx, self.gridpathy, c=np.arange(len(self.gridpathx)), cmap='Greys')
        self.visualise_update()

    def visualise_update(self):
        plt.pause(0.05)
        plt.draw()
