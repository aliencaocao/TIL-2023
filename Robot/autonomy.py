import time

from tilsdk import *  # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage  # import optional useful things

from planner import MyPlanner

SIMULATOR_MODE = True  # Change to False for real robomaster

if SIMULATOR_MODE:
    from tilsdk.mock_robomaster.robot import Robot  # Use this for the simulator
    from mock_services import CVService, NLPService
else:
    from robomaster.robot import Robot  # Use this for real robot
    from cv_service import CVService
    from nlp_service import NLPService

# Setup logging in a nice readable format
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')
formatter = logging.Formatter(fmt='[%(levelname)s][%(asctime)s][%(name)s]: %(message)s', datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
loggers = [logging.getLogger(name) for name in [__name__, 'NLPService', 'CVService']]
logger = loggers[0]  # the main one
logger.name = 'Main'
for l in loggers:
    l.propagate = False
    l.addHandler(handler)
    l.setLevel(logging.DEBUG)

# TODO: move the models to the robot folder for finals
NLP_PREPROCESSOR_DIR = '../ASR/wav2vec2-conformer'
NLP_MODEL_DIR = '../ASR/wav2vec2-conformer.trt'
OD_CONFIG_DIR = '../CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/cascade_internimage_l_fpn_3x_coco_custom.py'
OD_MODEL_DIR = '../CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/InternImage-L epoch_12 stripped.pth'
REID_MODEL_DIR = '../CV/SOLIDER-REID/log_SGD_500epoch_continue_1e-4LR_expanded/transformer_21_map0.941492492396344_acc0.8535950183868408.pth'
REID_CONFIG_DIR = '../CV/SOLIDER-REID/TIL.yml'
prev_img_rpt_time = 0  # In global scope to allow convenient usage of global keyword in do_cv()


def main():
    # Connect to robot and start camera
    robot = Robot()
    robot.initialize(conn_type="sta")
    robot.camera.start_video_stream(display=False, resolution='720p')

    # Initialize services
    if SIMULATOR_MODE:
        cv_service = CVService(OD_CONFIG_DIR, OD_MODEL_DIR, REID_MODEL_DIR, REID_CONFIG_DIR)
        nlp_service = NLPService(NLP_PREPROCESSOR_DIR, NLP_MODEL_DIR)
        loc_service = LocalizationService(host='localhost', port=5566)  # for simulator
        rep_service = ReportingService(host='localhost', port=5566)  # only avail on simulator
    else:
        cv_service = CVService(OD_CONFIG_DIR, OD_MODEL_DIR, REID_MODEL_DIR, REID_CONFIG_DIR)
        nlp_service = NLPService(NLP_PREPROCESSOR_DIR, NLP_MODEL_DIR)
        loc_service = LocalizationService(host='192.168.20.56', port=5521)  # for real robot
        # No rep_service needed for real robot

    # Initialize variables
    seen_clues = set()
    curr_loi: RealLocation = None
    path: List[RealLocation] = []
    lois: List[RealLocation] = []
    maybe_lois: List[RealLocation] = []
    curr_wp: RealLocation = None
    pose_filter = SimpleMovingAverage(n=10)
    map_: SignedDistanceGrid = loc_service.get_map()

    # Define helper functions
    # Filter function to exclude clues seen before   
    new_clues = lambda c: c.clue_id not in seen_clues

    # To update locations of interest, ignoring those seen before.
    def update_locations(old: List[RealLocation], new: List[RealLocation]) -> None:
        if new:
            for loc in new:
                if loc not in old:
                    logging.getLogger('update_locations').info('New location of interest: {}'.format(loc))
                    old.append(loc)

    # To run CV inference and report targets found
    def do_cv():
        global prev_img_rpt_time
        # print('pirt',prev_img_rpt_time)
        if not prev_img_rpt_time or time.time() - prev_img_rpt_time >= 1:  # throttle to 1 submission per second, and only read img if necessary
            img = robot.camera.read_cv2_image(strategy='newest')

            # Process image and detect targets
            targets = cv_service.targets_from_image(img)

            # Submit targets
            if targets:
                prev_img_rpt_time = time.time()
                rep_service.report(pose, img, targets)
                logger.info('{} targets detected.'.format(len(targets)))

    # Movement-related config and controls
    REACHED_THRESHOLD_M = 0.3  # TODO: Participant may tune, in meters
    ANGLE_THRESHOLD_DEG = 25.0  # TODO: Participant may tune.
    tracker = PIDController(Kp=(0.35, 0.2), Ki=(0.1, 0.0), Kd=(0, 0))

    # To prevent bug with endless spinning in alternate directions by only allowing 1 direction of spinni
    use_spin_direction_lock = False
    spin_direction_lock = False
    spin_sign = 0  # -1 or 1 when spin_direction_lock is active

    # To detect stuck and perform unstucking. New, needs IRL testing 
    use_stuck_detection = True
    log_x = []
    log_y = []
    log_time = []
    stuck_threshold_time_s = 15 # Minimum seconds to be considered stuck
    stuck_threshold_area_m = 0.15  # Considered stuck if it stays within a 15cm*15cm square

    # Initialise planner
    # Planner-related config here
    ROBOT_RADIUS_M = 0.17  # TODO: Participant may tune. 0.390 * 0.245 (L x W)
    map_ = map_.dilated(1.5 * ROBOT_RADIUS_M / map_.scale)

    planner = MyPlanner(map_,
                        waypoint_sparsity_m=0.4,
                        astargrid_threshold_dist_cm=29,
                        path_opt_min_straight_deg=165,
                        path_opt_max_safe_dist_cm=24,
                        explore_consider_nearest=4,
                        biggrid_size_m=0.8)

    # Start run
    rep_service.start_run()

    # Main loop
    while True:
        if path: planner.visualise_update()  # just for visualisation

        # Get new data
        pose, clues = loc_service.get_pose()
        pose = pose_filter.update(pose)
        pose = RealPose(min(pose[0], 7), min(pose[1], 5), pose[2])  # prevents out of bounds errors
        pose = RealPose(max(pose[0], 0), max(pose[1], 0), pose[2])  # prevents out of bounds errors
        if not pose:
            # no new data, continue to next iteration.
            continue

        # Set current location visit value to 1 if it is 0. Will not set it to 2 or higher values
        planner.visit(pose[:2])

        # Filter out clues that were seen before
        clues = list(filter(new_clues, clues))

        # Process clues using NLP and determine any new locations of interest
        if clues:
            new_lois, new_maybe_lois = nlp_service.locations_from_clues(clues)  # new locations of interest 
            if len(new_lois):
                logger.info('New location(s) of interest: {}.'.format(new_lois))
            update_locations(lois, new_lois)
            update_locations(maybe_lois, new_maybe_lois)
            seen_clues.update([c.clue_id for c in clues])

        # do_cv() # Debug

        # Reached current destination OR just started. Get next location of interest (i.e. destination to visit)
        if not curr_loi:

            # If no locations of interests from clues left,
            # Firstly visit those deemed fake clues by NLP service (incase NLP service was wrong)
            # Then explore the arena

            #TODO: choose the nearest one based on planned path length instead of Euclidean distance
            if len(lois) == 0:
                logger.info('No more locations of interest.')
                if len(maybe_lois): 
                    logger.info('Getting first of {} maybe_lois.'.format(len(maybe_lois)))
                    maybe_lois.sort(key=lambda l: euclidean_distance(l, pose), reverse=True)
                    nearest_maybe = maybe_lois.pop()
                    lois.append(nearest_maybe)
                else:
                    logger.info('Exploring the arena')
                    explore_next = planner.get_explore(pose[:2], debug=False)  # Debug plt is currently broken
                    print("Expl next", explore_next)
                    if explore_next is None:
                        break
                    lois.append(explore_next)
            else:  # >=1 LOI, sort to find the nearest one euclidean as heuristic.
                lois.sort(key=lambda l: euclidean_distance(l, pose), reverse=True)

            # Get new LOI
            curr_loi = lois.pop()
            logger.info('Current LOI set to: {}'.format(curr_loi))

            # Plan a path to the new LOI
            logger.info('Planning path to: {}'.format(curr_loi))
            path = planner.plan(pose[:2], curr_loi, display=True)

            if path is None:
                logger.info('No possible path found, location skipped')
                curr_loi = None #TODO: Make sure planner get_explore() is robust against enclosed area
            else:
                path.reverse()  # reverse so closest wp is last so that pop() is cheap , waypoint
                curr_wp = None
                logger.info('Path planned.')
        else:
            # There is a current LOI objective.
            # Continue with navigation along current path.
            if path:
                # Get next waypoint
                if not curr_wp:
                    curr_wp = path.pop()
                    logging.getLogger('Navigation').info('New waypoint: {}'.format(curr_wp))
                    if use_stuck_detection:  # Reset lists
                        log_x.clear()
                        log_y.clear()
                        log_time.clear()

                # Log location (for stuck detection purpose), delete old logs
                if use_stuck_detection:
                    log_x.append(pose[0])
                    log_y.append(pose[1])
                    now = time.time()
                    log_time.append(now)

                    # Remove records from more than 5 seconds before the time window examined to determine if stuck
                    while len(log_time) and log_time[0] < now - (stuck_threshold_time_s + 5):
                        log_time.pop(0)  # Technically O(N^2) but shouldn't matter due to small n (<100)
                        log_x.pop(0)
                        log_y.pop(0)

                    # assert len(log_time) == len(log_x) == len(log_y)
                    # print(len(log_time)) It stabilises around 80 in the simulator for threshold = 10s

                    # Stuck detection: Stuck if the robo is within a /0.15/m*/0.15/m box for the past /15/-/20/ seconds
                    if ((log_time[0] < now - stuck_threshold_time_s)
                            and (max(log_x) - min(log_x) <= stuck_threshold_area_m) \
                            and (max(log_y) - min(log_y) <= stuck_threshold_area_m)):
                        # Stuck! Try to unstuck by driving backwards at 0.5m/s for 2s.
                        # Then continue to next iteration for simplicity of code
                        print("STUCK DETECTED, DRIVING BACKWARDS")
                        robot.chassis.drive_speed(x=-0.3, z=0)
                        time.sleep(3)
                        robot.chassis.drive_speed(x=0, z=0)
                        tracker.reset()
                        continue

                # Calculate distance and heading to waypoint
                dist_to_wp = euclidean_distance(pose, curr_wp)
                ang_to_wp = np.degrees(np.arctan2(curr_wp[1] - pose[1], curr_wp[0] - pose[0]))
                ang_diff = -(ang_to_wp - pose[2])  # body frame

                # ensure ang_diff is in [-180, 180]
                if ang_diff < -180:
                    ang_diff += 360

                if ang_diff > 180:
                    ang_diff -= 360

                if use_spin_direction_lock and spin_direction_lock:  # disabled
                    if spin_sign == 1 and ang_diff < 0:
                        print("Spin direction lock, modifying ang_diff +360")
                        ang_diff += 360
                    elif spin_sign == -1 and ang_diff > 0:
                        print("Spin direction lock, modifying ang_diff -360")
                        ang_diff -= 360

                # logging.getLogger('Navigation').info('ang_to_wp: {}, hdg: {}, ang_diff: {}'.format(ang_to_wp, pose[2], ang_diff))
                # logging.getLogger('Navigation').info('Pose: {}'.format(pose))

                # Consider waypoint reached if within a threshold distance
                if dist_to_wp < (REACHED_THRESHOLD_M / 2 if len(path) <= 1 else REACHED_THRESHOLD_M):
                    logging.getLogger('Navigation').info('Reached wp: {}'.format(curr_wp))
                    tracker.reset()
                    curr_wp = None
                    continue

                # Determine velocity commands given distance and heading to waypoint
                vel_cmd = tracker.update((dist_to_wp, ang_diff))

                # logging.getLogger('Navigation').info('dist: {} ang:{} vel:{}'.format(dist_to_wp,ang_diff,vel_cmd))

                # reduce x velocity
                # Pose: (x, y, angle)
                # Vel_cmd: (speed, angle)
                vel_cmd[0] *= np.cos(np.radians(ang_diff))

                # If robot is facing the wrong direction, turn to face waypoint first before moving forward.
                # Lock spin direction (has effect only if use_spin_direction_lock = True) as bug causing infinite spinning back and forth has been encountered before in the sim
                if abs(ang_diff) > ANGLE_THRESHOLD_DEG:
                    spin_direction_lock = True
                    spin_sign = np.sign(ang_diff)
                    vel_cmd[0] = 0.0
                else:
                    spin_direction_lock = False
                    spin_sign = 0

                # Send command to robot
                robot.chassis.drive_speed(x=vel_cmd[0], z=vel_cmd[1])

            else:
                print('End of path. Spinning now.')
                curr_loi = None

                starting_angle = pose[2]
                starting_angle %= 360
                first_turn_angle = starting_angle % 45

                robot.chassis.drive_speed(x=0, z=first_turn_angle)
                time.sleep(1)
                robot.chassis.drive_speed(x=0, z=0)
                print("First_turn_angle", first_turn_angle)

                current_angle = (starting_angle - first_turn_angle) % 360

                print("Doing angle", current_angle)
                time.sleep(2)
                do_cv()

                for spinning in range(7):
                    robot.chassis.drive_speed(x=0, z=45)
                    time.sleep(1)
                    robot.chassis.drive_speed(x=0, z=0)
                    current_angle = (current_angle - 45) % 360

                    print("Doing angle", current_angle)
                    time.sleep(2)
                    do_cv()

                print('Done spinning. Moving on.')
                # Reset the pose_filter
                pose_filter = SimpleMovingAverage(n=10)
                continue

    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logger.info('Mission Terminated.')


if __name__ == '__main__':
    main()
