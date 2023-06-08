import math
import time

import pyastar2d
from skimage.draw import line as bresenham_sk
from tilsdk.localization import *


class MyPlanner:
    def __init__(self, map_: SignedDistanceGrid = None, waypoint_sparsity_m=0.5, astargrid_threshold_dist_cm=29, path_opt_min_straight_deg=170, path_opt_max_safe_dist_cm=24, ROBOT_RADIUS_M=0.17):
        '''
        Parameters
        ----------
        map : SignedDistanceGrid
            Distance grid map
        sdf_weight: float
            Relative weight of distance in cost function.
        waypoint_sparsity_m: float
            0.5 results in every 50th waypoint being initially taken at scale=0.01 and 10th at scale=0.05
        astargrid_threshold_dist_cm:
            Grid squares more than this distance away from a wall will be considered equivalent to this distance when calculating cost
        path_opt_min_straight_deg:
            After taking every nth waypoint (as controlled by waypoint_sparsity_m),
            but before being fed to the main program, 
            the path is optimised by removing waypoints in the middle of straight line.
            This value is the minimum degree formed by 3 consecutive points that we still consider 'straight'.
        path_opt_max_safe_dist_cm:
            Maximum allowed minimum clearance along paths from one waypoint to the next and previous for it to be deleted in optimisation.
        '''
        # ALL grids (including big_grid) use [y][x] convention

        self.waypoint_sparsity_m = waypoint_sparsity_m
        self.astargrid_threshold_dist_cm = astargrid_threshold_dist_cm
        self.path_opt_min_straight_deg = path_opt_min_straight_deg
        self.path_opt_max_safe_dist_cm = path_opt_max_safe_dist_cm

        self.map = map_
        self.passable = self.map.grid > 0
        self.bgrid = self.transform_add_border(self.map.grid.copy())  # Grid which takes the walls outside the grid into account
        self.bgrid -= 1.5 * ROBOT_RADIUS_M / self.map.scale  # Same functionality as .dilated last year: expands the walls by 1.5 times the radius of the robot
        self.astar_grid = self.transform_for_astar(self.bgrid.copy())

        self.path = None
        self.plt_init = False  # Whether the path visualisation pyplot for debug has been initialised
        self.scat = None  # For storing the scatterplot pyplot for path visualisation

    @staticmethod
    def transform_add_border(og_grid):
        grid = og_grid.copy()
        a, b = grid.shape
        for i in range(a):
            for j in range(b):
                grid[i][j] = min(grid[i][j], i + 1, a - i, j + 1, b - j)
                # The unit is pixels, so it is independent of the scale
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
                    # Convert unit of grid2 from pixel to cm
                    grid2[i][j] *= self.map.scale / 0.01
                    # We set areas further than 29cm away from walls/border to be of equal cost so that the robot can take a straight path when far from walls.
                    # The map is dilated 25.5cm (ROBOT_RADIUS_M=0.17),
                    # So a total clearance of 54c.5m shld be a very safe distance compared to robot length of 39cm/half length of 19.5cm.
                    grid2[i][j] = 1 + k / min(grid2[i][j], self.astargrid_threshold_dist_cm)
                else:
                    grid2[i][j] = np.inf
        return grid2.astype("float32")

    def nearest_clear(self, loc, passable):
        '''Utility function to find the nearest clear cell to a blocked cell'''
        loc = min(loc[0], self.map.grid.shape[0] - 1), min(loc[1], self.map.grid.shape[1] - 1)
        loc = max(loc[0], 0), max(loc[1], 0)
        if not passable[loc]:
            best = (1e18, (-1, -1))
            for i in range(self.map.height):  # y
                for j in range(self.map.width):  # x
                    if self.map.grid[(i, j)] > 0:
                        best = min(best, (euclidean_distance(GridLocation(i, j), loc), (i, j)))
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
        self.path = self.path + self.path[-1:]  # Duplicate destination wp to avoid bug in main loop which happens for the final waypoint as the path list is empty
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
        coeff = int(self.waypoint_sparsity_m / self.map.scale)  # default sparsity 0.5 = 50cm --> 50 for 0.01, 10 for 0.05
        path = list(path)
        if whole_path:
            path = [(x[1], x[0]) for x in path]  # Before: x,y; after: y,x
            return path
        else:
            coeff = max(coeff, 1)
            path = path[:1] + path[:-1:coeff] + path[-1:]  # Take the 1st, last, and every nth waypoint in the middle
            # Duplication of last waypoint to accomodate how main loop works has been moved to plan()
            path = self.optimize_path(path)
            path = [(x[1], x[0]) for x in path]  # Before: x,y; after: y,x
            return path

    def min_clearance_along_path(self, i1, j1, i2, j2):
        min_clearance = 1000
        # print('blist:', np.column_stack(bresenham_sk(i1, j1, i2, j2)))
        for i, j in np.column_stack(bresenham_sk(i1, j1, i2, j2)):
            min_clearance = min(min_clearance, self.bgrid[i][j] * self.map.scale / 0.01)
        return min_clearance

    def optimize_path(self, path: List[GridLocation]) -> List[GridLocation]:
        new_path = [path[0]]  # starting point always in path
        for i in range(1, len(path) - 1, 1):
            mcap = min(self.min_clearance_along_path(new_path[-1][0], new_path[-1][1], path[i][0], path[i][1]),
                       self.min_clearance_along_path(path[i + 1][0], path[i + 1][1], path[i][0], path[i][1]))
            # print("MCAP:",mcap)

            d1 = (path[i][0] - new_path[-1][0], path[i][1] - new_path[-1][1])
            d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            d1_deg = math.degrees(math.atan2(d1[0], d1[1]))
            d2_deg = math.degrees(math.atan2(d2[0], d2[1]))
            deg_diff = abs(d1_deg - d2_deg)
            deg_diff = min(deg_diff, 360 - deg_diff)
            # print("ANGLES: ", d1_deg, d2_deg, "DEG DIFF:", deg_diff)
            deg_tolerance = 180 - self.path_opt_min_straight_deg

            if mcap <= self.path_opt_max_safe_dist_cm and deg_diff > deg_tolerance:
                new_path.append(path[i])
            else:
                pass  # Skip this point if the angle formed with the next and prev points is more than 180 - X degrees OR the whole path >29cm from any wall
        new_path.append(path[-1])  # add last point
        return new_path

    def visualise_path(self):
        if not self.plt_init:
            pathmap = np.log(self.astar_grid)
            im = plt.imshow(pathmap)
            bar = plt.colorbar(im, extend='max')
            plt.title("Path: white -> start, black -> end.\nColorbar shows log(astar_grid) vals.")
            self.scat = plt.scatter(self.gridpathx, self.gridpathy, c=np.arange(len(self.gridpathx)), s=25, cmap='Greys')
            self.plt_init = True
        else:
            self.scat.remove()
            self.scat = plt.scatter(self.gridpathx, self.gridpathy, c=np.arange(len(self.gridpathx)), s=25, cmap='Greys')
        self.visualise_update()

    @staticmethod
    def visualise_update():
        plt.pause(0.05)
        plt.draw()
        # plt.savefig(f"path_{str(uuid.uuid4())[:5]}.png")

    def wall_clearance(self, l: Union[RealLocation, RealPose]):
        grid_l = self.map.real_to_grid(l)
        return self.bgrid[grid_l[1]][grid_l[0]] * self.map.scale
    
    def min_clearance_along_path_real(self, l1: Union[RealLocation, RealPose], l2: Union[RealLocation, RealPose]):
        """Returns the shortest distance to the nearest wall from the straight line from l1 and l2. Result in CM"""
        j1, i1 = self.map.real_to_grid(l1)[:2]
        j2, i2 = self.map.real_to_grid(l2)[:2]
        return self.min_clearance_along_path(i1, j1, i2, j2)


if __name__ == '__main__':
    print("Running (not importing) planner.py, doing debug")
    loc_service = LocalizationService(host='localhost', port=5566)  # for simulator
    map_: SignedDistanceGrid = loc_service.get_map()
    print("Got map from loc")
    planner = MyPlanner(map_, waypoint_sparsity_m=0.4, astargrid_threshold_dist_cm=29, path_opt_max_safe_dist_cm=24, path_opt_min_straight_deg=165, ROBOT_RADIUS_M=0.17)


    def test_path(a, b, c, d):
        path = planner.plan(start=RealLocation(a, b), goal=RealLocation(c, d), whole_path=False, display=True)


    # test_path(1,1,5,1)
    # plt.savefig("280523_o24_k500.png")

    for i in range(50):
        a = np.random.uniform(0.0, 7.0)
        b = np.random.uniform(0.0, 7.0)
        c = np.random.uniform(0.0, 7.0)
        d = np.random.uniform(0.0, 7.0)
        test_path(a, b, c, d)
        time.sleep(2)

    # im = plt.imshow(planner.astar_grid)
    # bar = plt.colorbar(im)
    # plt.show()

    # Random stuff below so that can set breakpoint right above for debugging
    # abc=3
    # k = 3

    # k += abc
    # print(k)
