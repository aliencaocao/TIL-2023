import glob
import os
import time
import warnings
warnings.filterwarnings("ignore")

from tilsdk import *  # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage  # import optional useful things

from planner import MyPlanner

SIMULATOR_MODE = True  # Change to False for real robomaster

import cv2

if SIMULATOR_MODE:
    from tilsdk.mock_robomaster.robot import Robot  # Use this for the simulator
    from mock_services import CVService, ASRService, SpeakerIDService
else:
    from robomaster.robot import Robot  # Use this for real robot
    from cv_service import CVService
    from nlp_service import ASRService, SpeakerIDService

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')
formatter = logging.Formatter(fmt='[%(levelname)s][%(asctime)s][%(name)s]: %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler('log.txt')
file_handler.setFormatter(formatter)
loggers = [logging.getLogger(name) for name in [__name__, 'NLPService', 'CVService', 'Navigation']]
logger = loggers[0]  # the main one
NavLogger = loggers[3]  # the navigation one
logger.name = 'Main'
for l in loggers:
    l.propagate = False
    l.addHandler(stream_handler)
    l.addHandler(file_handler)
    l.setLevel(logging.DEBUG)

# TODO: update the paths of imgs here. Model path are already simulated
suspect_img = cv2.imread('data/imgs/suspect1.png')
hostage_img = cv2.imread('data/imgs/targetmario.png')
ASR_MODEL_DIR = '../ASR/wav2vec2-conformer'
OD_CONFIG_PATH = '../CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/cascade_internimage_l_fpn_3x_coco_custom.py'
OD_MODEL_PATH = '../CV/InternImage/detection/work_dirs/cascade_internimage_l_fpn_3x_coco_custom/InternImage-L epoch_12 stripped.pth'
REID_MODEL_PATH = '../CV/SOLIDER-REID/log_SGD_500epoch_continue_1e-4LR_expanded/transformer_21_map0.941492492396344_acc0.8535950183868408.pth'
REID_CONFIG_PATH = '../CV/SOLIDER-REID/TIL.yml'
SPEAKERID_RUN_DIR = '../SpeakerID/m2d/evar/logs/til_ar_m2d.AR_M2D_cb0a37cc'
SPEAKERID_MODEL_FILENAME = 'weights_ep0it1-0.00000_loss0.1096.pth' # this is a FILENAME, not a full path
SPEAKERID_CONFIG_PATH = '../SpeakerID/m2d/evar/config/m2d.yaml'
ZIP_SAVE_DIR = Path("data/temp").absolute()
prev_img_rpt_time = 0  # In global scope to allow convenient usage of global keyword in do_cv()
robot = Robot()


def main():
    # Connect to robot
    robot.initialize(conn_type="ap")
    robot.set_robot_mode(mode="chassis_lead")

    # Initialize services
    cv_service = CVService(OD_CONFIG_PATH, OD_MODEL_PATH, REID_MODEL_PATH, REID_CONFIG_PATH)
    asr_service = ASRService(ASR_MODEL_DIR)
    speakerid_service = SpeakerIDService(SPEAKERID_CONFIG_PATH, SPEAKERID_RUN_DIR, SPEAKERID_MODEL_FILENAME)
    if SIMULATOR_MODE:
        loc_service = LocalizationService(host='localhost', port=5566)  # for simulator
        rep_service = ReportingService(host='localhost', port=5501)
    else:
        import torch
        print(torch.cuda.memory_summary())
        del torch
        loc_service = LocalizationService(host='192.168.20.56', port=5521)  # need change on spot
        rep_service = ReportingService(host='localhost', port=5566)  # need change on spot

    # Initialize variables
    curr_loi: RealLocation = None
    prev_loi: RealLocation = None
    target_rotation: float = None
    path: List[RealLocation] = []
    prev_wp: RealLocation = None
    curr_wp: RealLocation = None
    pose_filter = SimpleMovingAverage(n=10)
    map_: SignedDistanceGrid = loc_service.get_map()  # Currently, it is in the same format as the 2022 one. The docs says it's a new format.

    # If they update get_map() to match the description in the docs, we will need to write a function to convert it back to the 2022 format.

    # Movement-related config and controls
    REACHED_THRESHOLD_M = 0.3  # Participant may tune, in meters
    REACHED_THRESHOLD_LAST_M = REACHED_THRESHOLD_M / 2
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
    planner = MyPlanner(map_,
                        waypoint_sparsity_m=0.4,
                        astargrid_threshold_dist_cm=29,
                        path_opt_min_straight_deg=165,
                        path_opt_max_safe_dist_cm=24,
                        ROBOT_RADIUS_M = 0.17)  # Participant may tune. 0.390 * 0.245 (L x W)

    logger.info(f"Warming up pose filter to initialise position + reduce initial noise.")
    for _ in range(15):
        pose = loc_service.get_pose()
        time.sleep(0.25)
        pose = pose_filter.update(pose)

    # Start run
    res = rep_service.start_run()
    if res.status == 200:
        initial_target_pose = eval(res.data)
        logger.info(f"First location: {initial_target_pose}")
        curr_loi = RealLocation(x=initial_target_pose[0], y=initial_target_pose[1])
        target_rotation = initial_target_pose[2]
        path = planner.plan(pose[:2], curr_loi, display=True)
        if path is None:
            logger.info('Error, no possible path found to the first location!')
            #It should never come to this!
            #Re-initialise the map and planner with a smaller dilation (smaller robo radius) which I've done below (untested)
            map_.grid += 1.5 * ROBOT_RADIUS_M / map_.scale # Same functionality as .dilated last year: expands the walls by 1.5 times the radius of the robo
            ROBOT_RADIUS_M *= 2/3
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
        logger.error("Bad response from challenge server.")
        return

    def do_cv(pose: RealPose) -> str:
        global prev_img_rpt_time
        if not prev_img_rpt_time or time.time() - prev_img_rpt_time >= 1:  # throttle to 1 submission per second, and only read img if necessary
            robot.camera.start_video_stream(display=False, resolution='720p')
            if not SIMULATOR_MODE: print(robot.camera.conf)  # see if can see whitebalance values
            img = robot.camera.read_cv2_image(strategy='newest')
            robot.camera.stop_video_stream()
            img_with_bbox, answer = cv_service.predict([suspect_img, hostage_img], img)
            prev_img_rpt_time = time.time()
            return rep_service.report_situation(img_with_bbox, pose, answer, ZIP_SAVE_DIR)

    # Main loop
    while True:
        if path: planner.visualise_update()  # just for visualisation

        # Get new data
        pose = loc_service.get_pose()
        if not pose:
            # no new data, continue to next iteration.
            continue

        pose = pose_filter.update(pose)
        pose = RealPose(min(pose[0], 6.99), min(pose[1], 4.99), pose[2])  # prevents out of bounds errors
        pose = RealPose(max(pose[0], 0), max(pose[1], 0), pose[2])  # prevents out of bounds errors
        
        if not curr_loi:
            # We are at a checkpoint! Either the first one when just starting, or reached here after navigation.
            # Could be task or detour checkpoint or the end
            logger.info("Reached checkpoint")
            is_task_not_detour = None
            target_pose = None # Next location to go to which we'll receive soon
            info = rep_service.check_pose(pose)

            if isinstance(info, str): # Task checkpoint or the end
                if info == "End Goal Reached":  # end run and logs are after this loop, not implementing them here
                    break
                elif info == "Task Checkpoint Reached":
                    is_task_not_detour = True
                elif info == "Not An Expected Checkpoint":
                    logger.warning(f"Not yet at task checkpoint according to rep service but path planner thinks it is near enough. status: {res.status}, data: {res.data.decode()}, curr pose: {pose}")
                    # If we reached this execution branch, it means the autonomy code thinks the
                    # robot has reached close enough to the checkpoint, but the Reporting server
                    # is expecting the robot to be even closer to the checkpoint.
                    # Robot should try to get closer to the checkpoint.
                    
                    curr_loi = prev_loi  # Move closer to prev loi
                    path = [curr_loi, curr_loi] # 2 copies for legacy reasons... prob works with just 1 copy too but that would need testing
                    REACHED_THRESHOLD_LAST_M *= 2/3               
                    continue # Resume movement
                elif info == 'You Still Have Checkpoints':
                    logger.warning('Robot has reached end goal without finishing all task points!')  # TODO: what to do sia
                else:
                    raise ValueError(f"DSTA rep_service.check_pose() error: Unexpected string value: {info}.")
            elif isinstance(info, RealPose):  # Robot reached detour checkpoint and received new coordinates to go to.
                logger.info(f"Reached detour point, got next task point: {info}.")
                is_task_not_detour = False
                target_pose = info  # get the new task point from detour point
            else:
                raise Exception(f"DSTA rep_service.check_pose() error: Unexpected return type: {type(info)}.")
            if is_task_not_detour:
                logger.debug("Turning towards CV target")
                ang_diff = -(target_rotation - pose[2])  # body frame
                #Iinw the - sign is due to robomaster's angle convention being opposite of normal
                #If the below code results in wrong direction of rotation, try removing - sign

                # ensure ang_diff is in [-180, 180]
                if ang_diff < -180:
                    ang_diff += 360

                if ang_diff > 180:
                    ang_diff -= 360
                
                if ang_diff > 0:
                    robot.chassis.drive_speed(x=0, z=45)
                    time.sleep(ang_diff/45)
                    robot.chassis.drive_speed(x=0, z=0)
                else:
                    robot.chassis.drive_speed(x=0, z=-45)
                    time.sleep(ang_diff/-45)
                    robot.chassis.drive_speed(x=0, z=0)
                

                logger.info("Starting AI tasks")
                save_dir = do_cv(pose)  # reports the CV stuff and no matter the result, audios will be saved to ZIP_SAVE_DIR
                if not save_dir: continue  # still within 1second rate limit, skip to next iteration

                pred = speakerid_service.predict(glob.glob(os.path.join(save_dir, '*.wav')))
                pred = 'audio1_teamName One_member2' or pred  # if predict errored, it will return None, in this case just submit a dummy string and continue
                pred = 'audio1_teamName One_member2' # remove this line once speakerid is trained
                save_dir = rep_service.report_audio(pose, pred, ZIP_SAVE_DIR)

                # ASR password digits task
                password = asr_service.predict(glob.glob(os.path.join(save_dir, '*.wav')))
                if not password:
                    logger.warning('ASR failed to detect any digits in audio. Submitting (0,) as placeholder.')
                    password = (0,)
                target_pose = rep_service.report_digit(pose, password)
                target_pose_check = rep_service.check_pose(target_pose)
                if isinstance(target_pose_check, RealPose):
                    logger.info(f'Wrong password! Got detour point {target_pose}, moving there now')
                elif target_pose_check == 'Task Checkpoint Reached':
                    logger.info(f'Correct password! Moving to next task checkpoint at {target_pose}')
                elif target_pose_check == 'End Goal Reached':
                    logger.info(f'Correct password! Moving to final goal at {target_pose}')
                elif target_pose_check == 'You Still Have Checkpoints':
                    logger.warning('Correct password and target pose given is end goal, however you have missed some previous task checkpoints!')  # TODO: what to do sia
                else:
                    logger.error(f'Pose gotten from report digit is invalid! Got {target_pose}, check result {target_pose_check}.')

            curr_loi = RealLocation(target_pose[0], target_pose[1]) # Set the next location to be the target pose
            target_rotation = target_pose[2]

            logger.info('Next checkpoint location: {}'.format(curr_loi))
            # Reset the pose filter
            pose_filter = SimpleMovingAverage(n=10)

            # Plan a path to the new LOI
            logger.info('Planning path to: {}'.format(curr_loi))
            path = planner.plan(pose[:2], curr_loi, display=True)
            if path is None:
                logger.error('No possible path found to the next location!')
                # It should never come to this!
                # TODO: Implement some simple random movement for the robot to change its location

                # And/or re-initialise the map and planner with a smaller dilation (smaller robo radius) which I've done below (untested)
                planner = MyPlanner(map_,
                                    waypoint_sparsity_m=0.4,
                                    astargrid_threshold_dist_cm=29,
                                    path_opt_min_straight_deg=165,
                                    path_opt_max_safe_dist_cm=24,
                                    ROBOT_RADIUS_M = 0.17 * 2 / 3)
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
                    NavLogger.debug('New waypoint: {}'.format(curr_wp))
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
                        NavLogger.warning("STUCK DETECTED, DRIVING BACKWARDS")
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
                        NavLogger.debug("Spin direction lock, modifying ang_diff +360")
                        ang_diff += 360
                    elif spin_sign == -1 and ang_diff > 0:
                        NavLogger.debug("Spin direction lock, modifying ang_diff -360")
                        ang_diff -= 360

                # NavLogger.info('ang_to_wp: {}, hdg: {}, ang_diff: {}'.format(ang_to_wp, pose[2], ang_diff))
                # NavLogger.info('Pose: {}'.format(pose))

                # Consider waypoint reached if within a threshold distance
                if dist_to_wp < (REACHED_THRESHOLD_LAST_M if len(path) <= 1 else REACHED_THRESHOLD_M):
                    NavLogger.debug('Reached wp: {}'.format(curr_wp))
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
                # Calculate the angle threshold based on dist_to_wp such that when reaching the next wp, the max deviation is less than MAX_DEVIATION_THRESH_M , clamped to [5, 25] deg
                # MAX_DEVIATION_THRESH_M based on shortest distance to a wall from the straight line that connects to the next wp
                # This is only calculated at the first iter of every new wp and kept constant when approaching the same wp, to prevent angle threshold increasing too much when the robot is near the wp
                if prev_wp != curr_wp:
                    MAX_DEVIATION_THRESH_M = planner.min_clearance_along_path_real(pose, curr_wp) / 100  # returns cm so convert to m here
                    ANGLE_THRESHOLD_DEG = np.clip(np.degrees(np.arctan2(MAX_DEVIATION_THRESH_M, dist_to_wp)), 5, 25)  # TODO: tune the clamp range
                    NavLogger.debug(f'MAX_DEVIATION_THRESH_M: {MAX_DEVIATION_THRESH_M}, ANGLE_THRESHOLD_DEG: {ANGLE_THRESHOLD_DEG}')
                    prev_wp = curr_wp
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
                prev_loi = curr_loi
                curr_loi = None

    rep_service.end_run()  # Call this only after receiving confirmation from the scoring server that you have reached the maze's last checkpoint.
    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logger.info('Mission Completed.')
    if not SIMULATOR_MODE:
        robot.close()
        asr_service.language_tool.close()  # closes the spawned Java program, hangs on Windows (need task manager), need test on their Linux machine


if __name__ == '__main__':
    main()
