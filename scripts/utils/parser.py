import argparse


def setup_record_parser(
    subparsers: argparse.ArgumentParser, parent_parsers: list[argparse.ArgumentParser]
) -> argparse.ArgumentParser:
    """Setup parser for 'record' subcommand."""
    parser = subparsers.add_parser(
        "record",
        help="Record teleoperation data",
        parents=parent_parsers,
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )

    # Teleoperation Parameters
    parser.add_argument(
        "--teleop_device",
        type=str,
        default="keyboard",
        choices=["keyboard", "bi-keyboard", "so101leader", "bi-so101leader"],
        help="Device for interacting with environment",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Port for the teleop device:so101leader, default is /dev/ttyACM0",
    )
    parser.add_argument(
        "--left_arm_port",
        type=str,
        default="/dev/ttyACM0",
        help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
    )
    parser.add_argument(
        "--right_arm_port",
        type=str,
        default="/dev/ttyACM1",
        help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        default=False,
        help="recalibrate SO101-Leader or Bi-SO101Leader",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
    )
    # Task Configuration
    parser.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task.",
    )
    parser.add_argument(
        "--garment_name",
        type=str,
        default="Top_Long_Unseen_0",
        help="Name of the garment.",
    )
    parser.add_argument(
        "--garment_version", type=str, default="Release", help="Version of the garment."
    )
    parser.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of the garment configuration.",
    )
    parser.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path of the particle configuration.",
    )
    parser.add_argument(
        "--use_random_seed",
        action="store_true",
        default=False,
        help="Use random seed for the environment.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the environment."
    )
    parser.add_argument(
        "--log_success",
        action="store_true",
        default=False,
        help="Log success information.",
    )
    # Recording Parameters
    parser.add_argument(
        "--enable_record",
        action="store_true",
        default=False,
        help="Enable dataset recording function",
    )
    parser.add_argument(
        "--step_hz", type=int, default=120, help="Environment stepping rate in Hz."
    )
    parser.add_argument(
        "--num_episode",
        type=int,
        default=20,
        help="Maximum number of episodes to record",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Datasets/record",
        help="Root directory for saving recorded datasets (default: Datasets/record)",
    )
    parser.add_argument(
        "--disable_depth",
        action="store_true",
        default=False,
        help="Disable using top depth observation in env and dataset.",
    )
    parser.add_argument(
        "--enable_pointcloud",
        action="store_true",
        default=False,
        help="Whether to enable pointcloud observation in env and dataset.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help=" Description of the task to be performed.",
    )
    parser.add_argument(
        "--assist_policy_type",
        type=str,
        default=None,
        help="Optional policy type to run during recording before manual takeover.",
    )
    parser.add_argument(
        "--assist_policy_path",
        type=str,
        default=None,
        help="Policy checkpoint path for assisted recording.",
    )
    parser.add_argument(
        "--assist_dataset_root",
        type=str,
        default=None,
        help="Dataset root for assisted LeRobot policy metadata.",
    )
    parser.add_argument(
        "--assist_policy_device",
        type=str,
        default="cuda",
        help="Device for assisted policy inference (cuda or cpu).",
    )
    parser.add_argument(
        "--enable_click_ik",
        action="store_true",
        default=False,
        help=(
            "Enable assisted top-camera click-to-IK. In recording mode, press C/V "
            "to pause ACT and latch grippers, then click the top-camera window to "
            "move the nearest arm to that point."
        ),
    )
    parser.add_argument(
        "--scripted_oracle",
        type=str,
        default=None,
        choices=["auto", "top_short", "pant_long"],
        help=(
            "If set, run a scripted check_point-based fold oracle as the action "
            "source (replaces teleop and assist policy). 'auto' detects the "
            "category from the loaded garment. Recording is automatic: successful "
            "episodes are saved, failures are discarded and re-attempted."
        ),
    )
    parser.add_argument(
        "--click_ik_z_offset",
        type=float,
        default=0.05,
        help="Meters to lift the clicked depth point for the gripper IK target.",
    )
    parser.add_argument(
        "--click_ik_steps",
        type=int,
        default=45,
        help="Number of sim steps used to interpolate each click-to-IK move.",
    )
    parser.add_argument(
        "--ik_command_file",
        type=str,
        default="/tmp/lehome_ik_command.json",
        help=(
            "Optional live IK command JSON file polled by click-IK. "
            "This enables soft-pause + external command control without relying on the OpenCV GUI."
        ),
    )
    parser.add_argument(
        "--ik_status_file",
        type=str,
        default="/tmp/lehome_ik_status.json",
        help="Status JSON written after live IK commands are accepted or rejected.",
    )
    parser.add_argument(
        "--visual_repair_grasp",
        type=str,
        default="top",
        choices=["top", "bottom"],
        help=(
            "Landmark pair used by the E-key visual repair macro. "
            "'top' grasps shoulder/collar-side landmarks; 'bottom' grasps hem/leg-side landmarks."
        ),
    )
    parser.add_argument(
        "--auto_visual_repair_step",
        type=int,
        default=-1,
        help=(
            "If >=0, auto-start recording and trigger the E-key visual repair macro "
            "after this many episode steps. Useful for unattended repair-demo collection."
        ),
    )
    parser.add_argument(
        "--auto_visual_repair_attempts",
        type=int,
        default=3,
        help=(
            "Maximum number of closed-loop visual repair attempts per episode. "
            "After each attempt settles, the success checker decides whether to save "
            "or re-detect landmarks and try another repair."
        ),
    )
    parser.add_argument(
        "--auto_visual_repair_settle_steps",
        type=int,
        default=40,
        help="Settling steps after a visual repair before deciding whether to retry.",
    )
    parser.add_argument(
        "--visual_repair_debug_dir",
        type=str,
        default="/tmp/lehome_visual_repair_debug",
        help="Directory for top-camera debug PNGs saved during visual repair.",
    )
    parser.add_argument(
        "--visual_repair_debug_every",
        type=int,
        default=30,
        help="Save one visual-repair debug PNG every N macro steps; <=0 disables.",
    )
    parser.add_argument(
        "--auto_save_success",
        action="store_true",
        default=False,
        help="Automatically save an episode once the task success checker passes.",
    )
    parser.add_argument(
        "--auto_save_min_steps",
        type=int,
        default=120,
        help="Minimum episode steps before --auto_save_success may mark success.",
    )
    parser.add_argument(
        "--auto_save_success_settle_steps",
        type=int,
        default=0,
        help=(
            "Additional recording steps to wait after success and release gates "
            "first pass before auto-saving. The success/release checks must "
            "remain valid throughout this window; useful to let cloth land."
        ),
    )
    parser.add_argument(
        "--auto_save_require_release",
        action="store_true",
        default=False,
        help=(
            "When auto-saving successful rollouts, require both grippers to be "
            "open and optionally clear of the cloth. Useful for harvesting "
            "complete release/retract demonstrations instead of truncated folds."
        ),
    )
    parser.add_argument(
        "--auto_save_min_gripper_open",
        type=float,
        default=0.20,
        help="Minimum gripper joint value considered released for --auto_save_require_release.",
    )
    parser.add_argument(
        "--auto_save_min_gripper_cloth_distance",
        type=float,
        default=0.0,
        help=(
            "Minimum world-frame distance in meters from each gripper frame to "
            "the cloth before auto-save. <=0 disables the clearance check."
        ),
    )
    parser.add_argument(
        "--auto_start_record",
        action="store_true",
        default=False,
        help="Start recording automatically after the idle/stabilization phase.",
    )
    parser.add_argument(
        "--auto_restart_fail_steps",
        type=int,
        default=-1,
        help=(
            "If >=0, automatically discard/restart an episode after this many "
            "recording steps unless the success checker has already passed."
        ),
    )
    parser.add_argument(
        "--auto_save_near_miss",
        action="store_true",
        default=False,
        help=(
            "At score probes or --auto_restart_fail_steps, save a high-quality "
            "near-miss instead of discarding it. Use only into a separate "
            "weak/near-miss dataset."
        ),
    )
    parser.add_argument(
        "--auto_save_near_miss_min_passed",
        type=int,
        default=4,
        help="Minimum success-check conditions that must pass for near-miss saving.",
    )
    parser.add_argument(
        "--auto_save_near_miss_max_worst_close_ratio",
        type=float,
        default=1.10,
        help=(
            "Maximum worst ratio over <= success conditions for near-miss saving. "
            "Example: 1.10 allows a close-distance miss up to 10%% over threshold."
        ),
    )
    parser.add_argument(
        "--auto_save_near_miss_require_release",
        action="store_true",
        default=False,
        help="Require the same gripper release/clearance gate for near-miss saves.",
    )
    parser.add_argument(
        "--max_attempts_per_episode",
        type=int,
        default=-1,
        help=(
            "If >0, abort the current recording run after this many failed "
            "attempts for the same episode. Useful for unattended harvesters "
            "so one hard garment/variant cannot run forever."
        ),
    )
    parser.add_argument(
        "--early_restart_schedule",
        type=str,
        default="",
        help=(
            "Comma-separated staged early-restart rules. Format per rule is "
            "step:min_passed:best_close_ratio:min_close_passed. Empty fields "
            "disable that check. Example: '160:2:2.8:,220:3::2' restarts "
            "obvious misses at 160 and one-leg failures at 220."
        ),
    )
    parser.add_argument(
        "--auto_grip_hold_start_step",
        type=int,
        default=-1,
        help=(
            "If >=0, force both grippers closed starting at this recording step. "
            "Used for ACT harvesting when the policy releases before the fold is complete."
        ),
    )
    parser.add_argument(
        "--auto_grip_hold_end_step",
        type=int,
        default=-1,
        help="Recording step where the forced-closed gripper hold ends.",
    )
    parser.add_argument(
        "--auto_grip_release_until_step",
        type=int,
        default=-1,
        help=(
            "If > auto_grip_hold_end_step, force both grippers open from hold_end "
            "until this recording step, giving the episode an explicit release phase."
        ),
    )
    parser.add_argument(
        "--score_probe_steps",
        type=str,
        default="",
        help=(
            "Comma-separated recording steps at which to log raw cloth checkpoint "
            "positions/distances for later success-vs-failure comparison."
        ),
    )
    parser.add_argument(
        "--score_probe_log",
        type=str,
        default="",
        help="JSONL path for --score_probe_steps and terminal save/restart scores.",
    )
    parser.add_argument(
        "--early_restart_step",
        type=int,
        default=-1,
        help=(
            "If >=0, evaluate progress at this step and restart early when the "
            "fold is clearly hopeless."
        ),
    )
    parser.add_argument(
        "--early_restart_close_ratio",
        type=float,
        default=3.0,
        help=(
            "For <= success conditions, restart early if the best closing-condition "
            "ratio value/threshold is still above this value."
        ),
    )
    parser.add_argument(
        "--early_restart_min_passed",
        type=int,
        default=0,
        help="Minimum number of success-check conditions that must pass by early_restart_step.",
    )
    parser.add_argument(
        "--safe_assist_hotkeys",
        action="store_true",
        default=False,
        help=(
            "Disable destructive/manual assist hotkeys (N/D/X/M/E/G) while keeping "
            "S/P/R/C/V/Z/ESC. Useful for ACT-assisted recording with auto-save."
        ),
    )
    parser.add_argument(
        "--record_ee_pose",
        action="store_true",
        default=False,
        help="Record end-effector pose online (requires Pinocchio and scipy)",
    )
    parser.add_argument(
        "--ee_urdf_path",
        type=str,
        default=None,
        help="URDF file path (required only when using --record_ee_pose)",
    )
    parser.add_argument(
        "--ee_state_unit",
        type=str,
        default="rad",
        choices=["deg", "rad"],
        help="Joint angle unit for kinematic solver (default: rad)",
    )

    return parser


def setup_replay_parser(
    subparsers: argparse.ArgumentParser, parent_parsers: list[argparse.ArgumentParser]
) -> argparse.ArgumentParser:
    """Setup parser for 'replay' subcommand."""
    parser = subparsers.add_parser(
        "replay",
        help="Replay dataset",
        parents=parent_parsers,
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task environment.",
    )
    parser.add_argument(
        "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="Datasets/record/example/record_top_long_release_10/001",
        help="Root directory of the dataset to replay.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory to save replayed episodes (if None, replay only without saving).",
    )
    parser.add_argument(
        "--num_replays",
        type=int,
        default=1,
        help="Number of times to replay each episode.",
    )
    parser.add_argument(
        "--save_successful_only",
        action="store_true",
        default=False,
        help="Only save episodes that achieve success during replay.",
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Starting episode index (inclusive).",
    )
    parser.add_argument(
        "--end_episode",
        type=int,
        default=None,
        help="Ending episode index (exclusive). If None, replay all episodes.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help="Description of the task to be performed.",
    )
    parser.add_argument(
        "--garment_version", type=str, default="Release", help="Version of the garment."
    )
    parser.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path of the garment configuration.",
    )
    parser.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path of the particle configuration.",
    )
    parser.add_argument(
        "--use_ee_pose",
        action="store_true",
        default=False,
        help="Use action.ee_pose (Cartesian space) control, converted to joint angles via IK.",
    )
    parser.add_argument(
        "--ee_urdf_path",
        type=str,
        default="Assets/robots/so101_new_calib.urdf",
        help="URDF file path (required when using --use_ee_pose).",
    )
    parser.add_argument(
        "--ee_state_unit",
        type=str,
        default="rad",
        choices=["deg", "rad"],
        help="Joint angle unit for kinematic solver (default: rad).",
    )
    parser.add_argument(
        "--disable_depth",
        action="store_true",
        default=False,
        help="Disable depth observation during replay.",
    )

    return parser


def setup_inspect_parser(
    subparsers: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Setup parser for 'inspect' subcommand."""
    parser = subparsers.add_parser("inspect", help="Inspect dataset metadata")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--show_frames", type=int, default=None, help="Display first N frames"
    )
    parser.add_argument("--show_stats", action="store_true", help="Display statistics")
    return parser


def setup_read_parser(subparsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Setup parser for 'read' subcommand."""
    parser = subparsers.add_parser("read", help="Read dataset states")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--num_frames", type=int, default=None, help="Number of frames to read"
    )
    parser.add_argument(
        "--episode", type=int, default=None, help="Specific episode index"
    )
    parser.add_argument("--output_csv", type=str, default=None, help="Export to CSV")
    parser.add_argument("--show_stats", action="store_true", help="Display statistics")
    return parser


def setup_augment_parser(
    subparsers: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Setup parser for 'augment' subcommand."""
    parser = subparsers.add_parser("augment", help="Add end-effector pose to dataset")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument("--urdf_path", type=str, required=True, help="URDF file path")
    parser.add_argument(
        "--state_unit",
        type=str,
        default="rad",
        choices=["rad", "deg"],
        help="Joint angle unit",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output directory (default: in-place)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing EE pose data"
    )
    return parser


def setup_merge_parser(subparsers: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Setup parser for 'merge' subcommand."""
    parser = subparsers.add_parser("merge", help="Merge multiple datasets")
    parser.add_argument(
        "--source_roots",
        type=str,
        required=True,
        help="List of source dataset directories (as Python list string)",
    )
    parser.add_argument(
        "--output_root", type=str, required=True, help="Output dataset directory"
    )
    parser.add_argument(
        "--output_repo_id", type=str, default="merged_dataset", help="Repository ID"
    )
    parser.add_argument(
        "--merge_custom_meta",
        action="store_true",
        default=True,
        help="Merge custom meta files",
    )
    return parser


def setup_eval_parser() -> argparse.ArgumentParser:
    """Setup parser for evaluation script.

    Returns:
        The parser with evaluation arguments added.
    """
    parser = argparse.ArgumentParser(
        description="A script for evaluating policy in lehome manipulation environments."
    )

    # Core arguments
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=600,
        help="Maximum number of steps per evaluation episode.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="LeHome-BiSO101-Direct-Garment-v2",
        help="Name of the task.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run for each garment.",
    )
    parser.add_argument(
        "--step_hz", type=int, default=120, help="Environment stepping rate in Hz."
    )
    # Evaluation parameters
    parser.add_argument(
        "--use_random_seed",
        action="store_true",
        default=False,
        help="Use random seed for the environment.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the environment."
    )
    parser.add_argument(
        "--garment_type",
        type=str,
        default="top_long",
        choices=["top_long", "top_short", "pant_long", "pant_short", "custom"],
        help="Type of garments to evaluate.",
    )
    parser.add_argument(
        "--garment_cfg_base_path",
        type=str,
        default="Assets/objects/Challenge_Garment",
        help="Base path to the garment configuration files.",
    )
    parser.add_argument(
        "--particle_cfg_path",
        type=str,
        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
        help="Path to the particle configuration file.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="fold the garment on the table",
        help="Task description for VLA models (used in complementary_data).",
    )

    # Record parameters
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="If set, save evaluation episodes as video.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="outputs/eval_videos",
        help="Directory to save evaluation videos.",
    )
    parser.add_argument(
        "--trajectory_log_dir",
        type=str,
        default=None,
        help="If set, write per-step trajectory CSV (state, action, reward, success) to this dir.",
    )
    parser.add_argument(
        "--save_datasets",
        action="store_true",
        help="If set, save evaluation episodes dataset(only success).",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default="Datasets/eval",
        help="Path to save evaluation datasets.",
    )
    parser.add_argument(
        "--eval_list_override",
        type=str,
        default=None,
        help=(
            "Path to a .txt file listing garment names (one per line) to evaluate. "
            "Overrides the default per-category list; used for sweep mini-suites."
        ),
    )

    # Policy arguments for Imitation Learning (IL)
    # Note: Available policy types are dynamically loaded from PolicyRegistry
    parser.add_argument(
        "--policy_type",
        type=str,
        default="lerobot",
        help=(
            "Type of policy to use. Available policies are registered in PolicyRegistry. "
            "Built-in options: 'lerobot', 'sac', 'custom'. "
            "Participants can register their own policies using @PolicyRegistry.register('my_policy')."
        ),
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default="outputs/train/diffusion_fold_1/checkpoints/100000/pretrained_model",
        help="Path to the pretrained IL policy checkpoint.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="Path of the train dataset (for metadata).",
    )
    parser.add_argument(
        "--use_ee_pose",
        action="store_true",
        help="If set, policy outputs end-effector poses instead of joint angles. IK will be used to convert to joint angles.",
    )
    parser.add_argument(
        "--ee_urdf_path",
        type=str,
        default="Assets/robots/so101_new_calib.urdf",
        help="URDF path for IK solver (required when --use_ee_pose is set).",
    )
    parser.add_argument(
        "--replan_interval",
        type=int,
        default=0,
        help="Force ACT to re-plan every N steps (0 = use default chunk). "
             "Shorter intervals give tighter closed-loop control for deformable tasks.",
    )

    return parser
