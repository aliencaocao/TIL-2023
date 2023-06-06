import glob
import time

from tilsdk import *  # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage  # import optional useful things

from planner import MyPlanner

SIMULATOR_MODE = True  # Change to False for real robomaster

import cv2

if SIMULATOR_MODE:
    from tilsdk.mock_robomaster.robot import Robot  # Use this for the simulator
    from mock_services import CVService
else:
    from robomaster.robot import Robot  # Use this for real robot
    from cv_service import CVService
    from nlp_service import ASRService

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')
formatter = logging.Formatter(fmt='[%(levelname)s][%(asctime)s][%(name)s]: %(message)s', datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
loggers = [logging.getLogger(name) for name in [__name__, 'NLPService', 'CVService', 'Navigation']]
logger = loggers[0]  # the main one
NavLogger = loggers[3]  # the navigation one
logger.name = 'Main'
for l in loggers:
    l.propagate = False
    l.addHandler(handler)
    l.setLevel(logging.DEBUG)

# TODO: move the models to the robot folder for finals
ASR_PREPROCESSOR_DIR = '../ASR/wav2vec2-conformer'
ASR_MODEL_DIR = '../ASR/wav2vec2-conformer.trt'
OD_CONFIG_DIR = '../CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/cascade_internimage_l_fpn_3x_coco_custom.py'
OD_MODEL_DIR = '../CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/InternImage-L epoch_12 stripped.pth'
REID_MODEL_DIR = '../CV/SOLIDER-REID/log_SGD_500epoch_continue_1e-4LR_expanded/transformer_21_map0.941492492396344_acc0.8535950183868408.pth'
REID_CONFIG_DIR = '../CV/SOLIDER-REID/TIL.yml'
ZIP_SAVE_DIR = "D:/TIL-AI 2023/til-22-finals/til-23-finals/temp"
prev_img_rpt_time = 0  # In global scope to allow convenient usage of global keyword in do_cv()


def main():
    # Connect to robot and start camera
    robot = Robot()
    robot.initialize(conn_type="sta")
    robot.camera.start_video_stream(display=False, resolution='720p')

    # Initialize services
    if SIMULATOR_MODE:
        cv_service = CVService(OD_CONFIG_DIR, OD_MODEL_DIR, REID_MODEL_DIR, REID_CONFIG_DIR)
        asr_service = ASRService(ASR_PREPROCESSOR_DIR, ASR_MODEL_DIR)
        loc_service = LocalizationService(host='localhost', port=5566)  # for simulator
        rep_service = ReportingService(host='localhost', port=5566)
    else:
        cv_service = CVService(OD_CONFIG_DIR, OD_MODEL_DIR, REID_MODEL_DIR, REID_CONFIG_DIR)
        asr_service = ASRService(ASR_PREPROCESSOR_DIR, ASR_MODEL_DIR)
        loc_service = LocalizationService(host='192.168.20.56', port=5521)  # need change on spot
        rep_service = ReportingService(host='localhost', port=5566)  # need change on spot

    # Initialize variables
    curr_loi: RealLocation = None
    path: List[RealLocation] = []
    curr_wp: RealLocation = None
    pose_filter = SimpleMovingAverage(n=10)
    map_: SignedDistanceGrid = loc_service.get_map()  # Currently, it is in the same format as the 2022 one. The docs says it's a new format.

    # If they update get_map() to match the description in the docs, we will need to write a function to convert it back to the 2022 format.

    # Define helper functions
    # To run CV inference and report targets found

    def do_cv():
        global prev_img_rpt_time
        if not prev_img_rpt_time or time.time() - prev_img_rpt_time >= 1:  # throttle to 1 submission per second, and only read img if necessary
            img = robot.camera.read_cv2_image(strategy='newest')
            img_with_bbox, answer = cv_service.predict([suspect_img, hostage_img], img)
            prev_img_rpt_time = time.time()
            rep_service.report_situation(img_with_bbox, pose, answer, ZIP_SAVE_DIR)

    # Movement-related config and controls
    REACHED_THRESHOLD_M = 0.3  # Participant may tune, in meters
    ANGLE_THRESHOLD_DEG = 25.0  # Participant may tune.
    tracker = PIDController(Kp=(0.35, 0.2), Ki=(0.1, 0.0), Kd=(0, 0))

    # To prevent bug with endless spinning in alternate directions by only allowing 1 direction of spinning
    use_spin_direction_lock = False
    spin_direction_lock = False
    spin_sign = 0  # -1 or 1 when spin_direction_lock is active

    # To detect stuck and perform unstucking. New, needs IRL testing 
    use_stuck_detection = False
    log_x = []
    log_y = []
    log_time = []
    stuck_threshold_time_s = 15  # Minimum seconds to be considered stuck
    stuck_threshold_area_m = 0.15  # Considered stuck if it stays within a 15cm*15cm square

    # Initialise planner
    # Planner-related config here
    ROBOT_RADIUS_M = 0.17  # Participant may tune. 0.390 * 0.245 (L x W)
    map_.grid -= 1.5 * ROBOT_RADIUS_M / map_.scale  # Same functionality as .dilated last year: expands the walls by 1.5 times the radius of the robo

    planner = MyPlanner(map_,
                        waypoint_sparsity_m=0.4,
                        astargrid_threshold_dist_cm=29,
                        path_opt_min_straight_deg=165,
                        path_opt_max_safe_dist_cm=24)

    # Start run
    response = rep_service.start_run()  # This must be called before other ReportingService methods are called.

    # Main loop
    while True:
        if path: planner.visualise_update()  # just for visualisation

        # Get new data
        pose = loc_service.get_pose()
        pose = pose_filter.update(pose)
        pose = RealPose(min(pose[0], 7), min(pose[1], 5), pose[2])  # prevents out of bounds errors
        pose = RealPose(max(pose[0], 0), max(pose[1], 0), pose[2])  # prevents out of bounds errors
        if not pose:
            # no new data, continue to next iteration.
            continue

        # do_cv() # Debug

        if not curr_loi:
            # We are at a checkpoint! Either the first one when just starting, or reached here after navigation.

            return_val = "Test"  # rep_service.check_pose(pose) # Call this to check if the robot is currently at a task or detour checkpoint.

            if return_val == "PLACEHOLDER_FOR_TASKCHECKPT_VALUE":
                logger.debug('Spinning and taking photos to detect plushies')
                # do cv & spin 45 degrees 8 times
                starting_angle = pose[2]
                starting_angle %= 360
                first_turn_angle = starting_angle % 45

                robot.chassis.drive_speed(x=0, z=first_turn_angle)
                time.sleep(1)
                robot.chassis.drive_speed(x=0, z=0)
                logger.debug("First_turn_angle", first_turn_angle)

                current_angle = (starting_angle - first_turn_angle) % 360

                logger.debug("Doing angle", current_angle)
                time.sleep(2)
                do_cv()

                for spinning in range(7):
                    robot.chassis.drive_speed(x=0, z=45)
                    time.sleep(1)
                    robot.chassis.drive_speed(x=0, z=0)
                    current_angle = (current_angle - 45) % 360

                    logger.debug("Doing angle", current_angle)
                    time.sleep(2)
                    do_cv()

                logger.debug('Done spinning. Moving on.')

                # TODO: Code for speaker identification challenge

                # TODO: Code for ASR (decoding digits) challenge
                password = asr_service.predict(glob.glob(os.path.join(ZIP_SAVE_DIR, '*.wav')))
                if password:
                    target_pose = rep_service.report_digit(pose, password)
                    target_pose_check = rep_service.check_pose(target_pose)
                    if isinstance(target_pose_check, RealPose):  # TODO: check if legal
                        logger.info(f'Wrong password! Got detour point {target_pose}, next task point is {target_pose_check}, moving there now')
                    if target_pose_check == 'Task Checkpoint Reached':
                        logger.info(f'Correct password! Moving to next task checkpoint at {target_pose}')
                    elif target_pose_check == 'Goal Reached':
                        logger.info(f'Correct password! Moving to final goal at {target_pose}')
                    else:
                        logger.warning('Pose gotten from report digit is invalid!')

            # TODO: Get the coordinates for the next location
            # TODO: Break if signal is received that we've just completed the final checkpoint
            curr_loi = RealLocation(4, 1)  # Placeholder for getting the next location from wtv DSTA API

            logger.info('Current destination set to: {}'.format(curr_loi))

            # Reset the pose filter
            pose_filter = SimpleMovingAverage(n=10)

            # Plan a path to the new LOI
            logger.info('Planning path to: {}'.format(curr_loi))
            path = planner.plan(pose[:2], curr_loi, display=True)
            if path is None:
                logger.info('Catastrophic error, no possible path found!')
                # It should never come to this!
                # TODO: Implement some simple random movement for the robot to change its location

                # And/or re-initialise the map and planner with a smaller dilation (smaller robo radius) which I've done below (untested)
                map_.grid += 1.5 * ROBOT_RADIUS_M / map_.scale  # Same functionality as .dilated last year: expands the walls by 1.5 times the radius of the robo
                ROBOT_RADIUS_M *= 2 / 3
                map_.grid -= 1.5 * ROBOT_RADIUS_M / map_.scale
                planner = MyPlanner(map_,
                                    waypoint_sparsity_m=0.4,
                                    astargrid_threshold_dist_cm=29,
                                    path_opt_min_straight_deg=165,
                                    path_opt_max_safe_dist_cm=24)
            else:
                path.reverse()  # reverse so closest wp is last so that pop() is cheap , waypoint
                curr_wp = None
                logger.info('Path planned.')

        else:
            # There is a current destination.
            # Continue with navigation along current path.
            if path:  # [RealLocation(x,y)]
                # Get next waypoint
                if not curr_wp:
                    curr_wp = path.pop()
                    NavLogger.info('New waypoint: {}'.format(curr_wp))
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
                    # logger.debug(len(log_time)) It stabilises around 80 in the simulator for threshold = 10s

                    # Stuck detection: Stuck if the robo is within a /0.15/m*/0.15/m box for the past /15/-/20/ seconds
                    if ((log_time[0] < now - stuck_threshold_time_s)
                            and (max(log_x) - min(log_x) <= stuck_threshold_area_m)
                            and (max(log_y) - min(log_y) <= stuck_threshold_area_m)):
                        # Stuck! Try to unstuck by driving backwards at 0.5m/s for 2s.
                        # Then continue to next iteration for simplicity of code
                        logger.warning("STUCK DETECTED, DRIVING BACKWARDS")
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
                        logger.debug("Spin direction lock, modifying ang_diff +360")
                        ang_diff += 360
                    elif spin_sign == -1 and ang_diff > 0:
                        logger.debug("Spin direction lock, modifying ang_diff -360")
                        ang_diff -= 360

                # NavLogger.info('ang_to_wp: {}, hdg: {}, ang_diff: {}'.format(ang_to_wp, pose[2], ang_diff))
                # NavLogger.info('Pose: {}'.format(pose))

                # Consider waypoint reached if within a threshold distance
                if dist_to_wp < (REACHED_THRESHOLD_M / 2 if len(path) <= 1 else REACHED_THRESHOLD_M):
                    NavLogger.info('Reached wp: {}'.format(curr_wp))
                    tracker.reset()
                    curr_wp = None
                    continue

                # Determine velocity commands given distance and heading to waypoint
                vel_cmd = tracker.update((dist_to_wp, ang_diff))

                # NavLogger.info('dist: {} ang:{} vel:{}'.format(dist_to_wp,ang_diff,vel_cmd))

                # reduce x velocity
                # Pose: (x, y, angle)
                # Vel_cmd: (speed, angle)
                vel_cmd[0] *= np.cos(np.radians(ang_diff))

                # If robot is facing the wrong direction, turn to face waypoint first before moving forward.
                # Lock spin direction (has effect only if use_spin_direction_lock = True) as bug causing infinite spinning back and forth has been encountered before in the sim
                if abs(ang_diff) > ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0  # Robot can only rotate; set speed to 0
                    spin_direction_lock = True
                    spin_sign = np.sign(ang_diff)
                else:
                    spin_direction_lock = False
                    spin_sign = 0

                # Send command to robot
                robot.chassis.drive_speed(x=vel_cmd[0], z=vel_cmd[1])

            else:
                logger.debug('End of path.')
                curr_loi = None

    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    response = rep_service.end_run()  # Call this only after receiving confirmation from the scoring server that you have reached the maze's last checkpoint.
    logger.info('Mission Terminated.')


if __name__ == '__main__':
    suspect_img = cv2.imread('data/imgs/suspect1.png')
    hostage_img = cv2.imread('data/imgs/targetmario.png')
    main()
