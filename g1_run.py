from __future__ import annotations

import os, sys

import rclpy
from rclpy.node import Node
from rclpy.time import Time as rosTime

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from motion_reference_msgs.msg import MotionReference

import numpy as np
import torch
import yaml
import onnxruntime as ort

from unitree_ros2_real import UnitreeRos2Real

class G1Node(UnitreeRos2Real):
    def __init__(self,
            agent_cfg: dict = dict(),
            interested_link_idx: int = None, # the link index to visualize the transform
            forward_kinematics_freq: float = 100., # the frequency of forward kinematics computation
            default_motion_ref_length: int = None, # if given, override the value from env.yaml and ignore exceeding input.
            startup_step_size: float = 0.1, # the step size during startup stage
            wait_until_motion_ref_received: bool = True, # if True, the node will wait until the motion reference is received.
            model_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.agent_cfg = agent_cfg
        self.interested_link_idx = interested_link_idx
        self.forward_kinematics_freq = forward_kinematics_freq
        self.default_motion_ref_length = default_motion_ref_length
        self.startup_step_size = startup_step_size
        self.wait_until_motion_ref_received = wait_until_motion_ref_received
        self.model_device = model_device

        self._initialize_motion_reference_buffer()
    
    def _initialize_motion_reference_buffer(self):
        """ Initialize the motion reference buffer for the observation terms.
        Making a concatenated buffer so that the motion reference can be fed to the model faster.
        """

        # initialize the interested links for forward kinematics and link_ref command buffer
        self._interested_link_names = self.cfg["scene"]["motion_reference"]["link_of_interests"]
        # self._interested_link_indices = self._robot_kinematics_chain.get_frame_indices(*self._interested_link_names) # in the order of interested_link_names
        self._interested_link_pos_b = np.zeros((len(self._interested_link_names), 3), dtype= np.float32) # in the order of interested_link_names
        # You may add link quaternion later.
        print("Interested links: ", self._interested_link_names)
        
        motion_frame_size = 0
        self.motion_ref_term_slices = dict()
        for obs_term_name in self.cfg["observations"]["policy"].keys():
            if "time_to_target" == obs_term_name:
                motion_frame_size += 1
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-1, motion_frame_size)
            elif "time_from_ref_update" == obs_term_name:
                motion_frame_size += 1
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-1, motion_frame_size)
            elif "pose_ref" == obs_term_name:
                motion_frame_size += 6 # 3 for position, 3 for axis-angle
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-6, motion_frame_size)
            elif "pose_ref_mask" == obs_term_name:
                motion_frame_size += 4 # plane mask, height mask, orientation mask, heading mask
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-4, motion_frame_size)
            elif "dof_pos_ref" == obs_term_name:
                motion_frame_size += self.NUM_DOF
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-self.NUM_DOF, motion_frame_size)
            elif "dof_pos_err_ref" == obs_term_name:
                motion_frame_size += self.NUM_DOF
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-self.NUM_DOF, motion_frame_size)
            elif "dof_pos_mask" == obs_term_name:
                motion_frame_size += self.NUM_DOF
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-self.NUM_DOF, motion_frame_size)
            elif "link_pos_ref" == obs_term_name:
                motion_frame_size += len(self._interested_link_names) * 3
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-len(self._interested_link_names)*3, motion_frame_size)
            elif "link_pos_err_ref" == obs_term_name:
                motion_frame_size += len(self._interested_link_names) * 3
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-len(self._interested_link_names)*3, motion_frame_size)
            elif "link_ref_mask" == obs_term_name:
                motion_frame_size += len(self._interested_link_names)
                self.motion_ref_term_slices[obs_term_name] = slice(motion_frame_size-len(self._interested_link_names), motion_frame_size)
        
        # initialize the motion reference buffer
        self.current_state_motion_ref = np.zeros(motion_frame_size, dtype= np.float32)
        self.current_state_motion_ref[self.motion_ref_term_slices["pose_ref_mask"]] = 1.0
        self.current_state_motion_ref[self.motion_ref_term_slices["dof_pos_mask"]] = 1.0
        self.current_state_motion_ref[self.motion_ref_term_slices["link_ref_mask"]] = 1.0
        self.motion_ref_refreshed_time = None if self.wait_until_motion_ref_received else self.get_clock().now()
        self.time_to_target_when_refreshed = np.zeros(
            self.cfg["scene"]["motion_reference"]["num_frames"] if self.default_motion_ref_length is None \
                else self.default_motion_ref_length,
            dtype= np.float32,
        )
        self.motion_ref_buffer = np.zeros(
            (
                self.cfg["scene"]["motion_reference"]["num_frames"] if self.default_motion_ref_length is None else self.default_motion_ref_length,
                motion_frame_size,
            ),
            dtype= np.float32,
        )
        # create a tf broadcaster for debugging forward kinematics.
        if self.interested_link_idx is not None:
            self._insterested_link_tf_broadcaster = TransformBroadcaster(self)

        # setup self.stage to prevent the main loop from running
        self.run_stage = None

    """
    Callbacks to handle ROS-wise inputs.
    """

    @torch.no_grad()
    def _robot_forward_kinematics_callback(self):
        """ Compute the forward kinematics based on lowstate, using exported ONNXProgram. """
        input_name = self.onnx_sessions["forward_kinematics.onnx"].get_inputs()[0].name
        onnx_outputs = self.onnx_sessions["forward_kinematics.onnx"].run(
            None,
            {
                input_name: np.expand_dims(self.dof_pos_, axis= 0),
            }
        )
        self._interested_link_pos_b[:] = onnx_outputs[0].reshape(len(self._interested_link_names), 3)
        
        if hasattr(self, "_insterested_link_tf_broadcaster"):
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = "torso_link"
            tf_msg.child_frame_id = "interested_link"
            tf_msg.transform.translation.x = self._interested_link_pos_b[self.interested_link_idx, 0].item()
            tf_msg.transform.translation.y = self._interested_link_pos_b[self.interested_link_idx, 1].item()
            tf_msg.transform.translation.z = self._interested_link_pos_b[self.interested_link_idx, 2].item()
            self._insterested_link_tf_broadcaster.sendTransform(tf_msg)

    def _motion_reference_callback(self, msg: MotionReference):
        # update the motion reference buffer
        # NOTE: DONT use time from message, use the current time becuause the msg.header.stamp might
        # come from another machine, which is not synchronized with the current machine.
        self.motion_ref_refreshed_time = self.get_clock().now()
        self.get_logger().info("motion reference received", once=True)
        for frame_idx, motion_frame in enumerate(msg.data):
            if frame_idx >= self.motion_ref_buffer.shape[0]:
                break
            for term_name, term_slice in self.motion_ref_term_slices.items():
                if term_name == "time_to_target":
                    self.time_to_target_when_refreshed[frame_idx] = motion_frame.time_to_target
                    self.motion_ref_buffer[frame_idx, term_slice] = motion_frame.time_to_target
                elif term_name == "pose_ref":
                    self.motion_ref_buffer[frame_idx, term_slice.start] = motion_frame.position.x
                    self.motion_ref_buffer[frame_idx, term_slice.start+1] = motion_frame.position.y
                    self.motion_ref_buffer[frame_idx, term_slice.start+2] = motion_frame.position.z
                    self.motion_ref_buffer[frame_idx, term_slice.start+3] = motion_frame.axisangle.x
                    self.motion_ref_buffer[frame_idx, term_slice.start+4] = motion_frame.axisangle.y
                    self.motion_ref_buffer[frame_idx, term_slice.start+5] = motion_frame.axisangle.z
                elif term_name == "pose_ref_mask":
                    self.motion_ref_buffer[frame_idx, term_slice] = motion_frame.pose_mask
                elif term_name == "dof_pos_ref":
                    self.motion_ref_buffer[frame_idx, term_slice] = motion_frame.dof_pos - self.default_dof_pos
                elif term_name == "dof_pos_err_ref":
                    self.motion_ref_buffer[frame_idx, term_slice] = motion_frame.dof_pos - self.dof_pos_
                elif term_name == "dof_pos_mask":
                    self.motion_ref_buffer[frame_idx, term_slice] = motion_frame.dof_pos_mask
                elif term_name == "link_pos_ref":
                    link_pos = np.stack([
                        [lp.x, lp.y, lp.z] for lp in motion_frame.link_pos
                    ]).flatten()
                    self.motion_ref_buffer[frame_idx, term_slice] = link_pos
                elif term_name == "link_pos_err_ref":
                    # assuming `link_pos_ref` is already needed in current setting, so reuse `link_pos` variable
                    # computed by previous branch.
                    self.motion_ref_buffer[frame_idx, term_slice] = link_pos - self._interested_link_pos_b.flatten()
                elif term_name == "link_ref_mask":
                    self.motion_ref_buffer[frame_idx, term_slice] = motion_frame.link_pos_mask

    """
    Additional Observation Terms.
    """

    def _get_motion_ref_obs(self):
        """ Considering motion reference should be a connected buffer, using this may reduce memory copy.
        """
        # update time and local errors (pose depends on odometry, which is not updated here)
        time = self.get_clock().now() - self.motion_ref_refreshed_time
        time_passed_from_refreshed = time.nanoseconds / 1e-9
        self.motion_ref_buffer[:, self.motion_ref_term_slices["time_to_target"]] = (self.time_to_target_when_refreshed - time_passed_from_refreshed)[:, None]
        self.motion_ref_buffer[:, self.motion_ref_term_slices["time_from_ref_update"]] = time_passed_from_refreshed

        # update the motion reference buffer in the error terms
        self.motion_ref_buffer[:, self.motion_ref_term_slices["dof_pos_err_ref"]] = self.motion_ref_buffer[:, self.motion_ref_term_slices["dof_pos_ref"]] - self.dof_pos_[None, :]
        self.motion_ref_buffer[:, self.motion_ref_term_slices["link_pos_err_ref"]] = self.motion_ref_buffer[:, self.motion_ref_term_slices["link_pos_ref"]] - self._interested_link_pos_b.flatten()[None, :]

        # update the current state as pseudo-motion reference
        self.current_state_motion_ref[self.motion_ref_term_slices["dof_pos_ref"]] = self._get_dof_pos_obs()
        self.current_state_motion_ref[self.motion_ref_term_slices["link_pos_ref"]] = self._interested_link_pos_b.flatten()
        
        return np.concatenate([
            self.motion_ref_buffer,
            np.expand_dims(self.current_state_motion_ref, axis=0),
        ], axis= 0) # (num_frames+1, motion_frame_size)
    
    def _get_proprioception_obs(self, nolinvel: bool = True):
        """ Get the proprioception observation terms.
        NOTE: Currently when getting proprioception, obs clip is not implemented.
        Args:
            nolinvel: If True, the linear velocity will be ignored.
        Returns:
            obs: The proprioception observation terms. (no batchwise)
        """
        obs = []
        if not nolinvel:
            obs.append(self._get_lin_vel_obs() * self.obs_scales.get("base_lin_vel", 1.0))
        # NOTE!!! the observation names in training config file might be different from the names in
        # this node system. So, you should check the names in the training config file.
        obs.append(self._get_ang_vel_obs() * self.obs_scales.get("base_ang_vel", 1.0))
        obs.append(self._get_projected_gravity_obs() * self.obs_scales.get("projected_gravity", 1.0))
        obs.append(self._get_dof_pos_obs() * self.obs_scales.get("joint_pos", 1.0))
        obs.append(self._get_dof_vel_obs() * self.obs_scales.get("joint_vel", 1.0))
        obs.append(self._get_last_actions_obs() * self.obs_scales.get("last_action", 1.0))
        return np.concatenate(obs, axis= 0)

    """
    Interfaces for the main function to call and execute.
    """

    def start_ros_handlers(self):
        super().start_ros_handlers()

        if "forward_kinematics.onnx" in self.onnx_sessions and self.forward_kinematics_freq > 0:
            self.forward_kinematics_timer = self.create_timer(
                1./self.forward_kinematics_freq,
                self._robot_forward_kinematics_callback,
            )

        self.motion_reference_sub = self.create_subscription(
            MotionReference,
            "/motion_reference",
            self._motion_reference_callback,
            1,
        )

        main_loop_duration = self.cfg["sim"]["dt"] * self.cfg["decimation"]
        print("Starting main loop with duration: ", main_loop_duration)
        self.main_loop_timer = self.create_timer(main_loop_duration, self.main_loop)

    def register_network(self, onnx_sessions: dict):
        """Register the model to this node, and run the main loop."""
        self.onnx_sessions = onnx_sessions

    def main_loop(self):
        """The single loop for the robot to execute.
        NOTE: customized, might not generalize to other settings.
        """
        # Wait until the motion reference is received.
        if self.motion_ref_refreshed_time is None:
            self.get_logger().info("Waiting for the motion reference to be received.", throttle_duration_sec= 5)
            return
        
        # Run the network no matter which stage it is.
        # deal with motion reference encoder
        # NOTE: `_get_motion_ref_obs` must be called whenever you want to access `self.motion_ref_buffer`
        # because it contains some real-time information.
        motion_ref_obs = np.expand_dims(self._get_motion_ref_obs(), axis= 0)
        embeddings_input_name = self.onnx_sessions["motion_ref.onnx"].get_inputs()[0].name
        embeddings = self.onnx_sessions["motion_ref.onnx"].run(None, {embeddings_input_name: motion_ref_obs})[0]
        proprioception_obs = np.concatenate([
            np.expand_dims(self._get_proprioception_obs(), axis= 0),
            embeddings,
        ], axis= 1)
        # deal with GRU/MLP actor
        actor_inputs = self.onnx_sessions["actor.onnx"].get_inputs()
        if len(actor_inputs) > 1:
            # GRU actor
            actor_input_name = actor_inputs[0].name
            actor_hidden_name = actor_inputs[1].name
            if not hasattr(self, "actor_hidden"):
                self.actor_hidden = np.zeros((1, *actor_hidden_name.shape), dtype= np.float32)
            actions, actor_hidden = self.onnx_sessions["actor.onnx"].run(
                None, {actor_input_name: proprioception_obs, actor_hidden_name: self.actor_hidden},
            )
            if self.run_stage == "policy": # update hidden state only when running policy
                self.actor_hidden = actor_hidden
        else:
            # MLP actor
            actor_input_name = actor_inputs[0].name
            actions = self.onnx_sessions["actor.onnx"].run(
                None, {actor_input_name: proprioception_obs},
            )[0]

        # Send the action depending on the stage.
        if self.run_stage == "startup":
            """Set the action following the motion reference directly."""
            # num_frames, NUM_DOF
            dof_pos_target = self.motion_ref_buffer[0, self.motion_ref_term_slices["dof_pos_ref"]]
            dof_pos_err = self.motion_ref_buffer[0, self.motion_ref_term_slices["dof_pos_err_ref"]]
            err_large_mask = np.abs(dof_pos_err) > self.startup_step_size
            if not err_large_mask.any():
                self.get_logger().info("Joint close to 0-th frame of motion reference", throttle_duration_sec= 1)
            dof_pos_cmd = np.where(
                err_large_mask,
                self.dof_pos_ + np.sign(dof_pos_err) * self.startup_step_size,
                dof_pos_target,
            )
            actions = np.expand_dims(
                (dof_pos_cmd - self.default_dof_pos) / self.actions_scale,
                axis= 0,
            )
            self.send_action(actions[0])
        elif self.run_stage == "policy":
            """Run the policy network to get the action."""
            self.send_action(actions[0])
        else:
            # self.get_logger().warn("Unknown stage: " + str(self.run_stage), throttle_duration_sec= 0.2)
            pass

        # Switch the stage if necessary.
        if self.joy_stick_buffer.keys & self.WirelessButtons.R1:
            if self.run_stage == "policy":
                self.get_logger().warn(
                    "Should not switch to startup stage while running policy. No switching. Current stage: " + self.run_stage,
                    throttle_duration_sec= 0.2,
                )
            else:
                self.run_stage = "startup"
                self.get_logger().info("Switched to startup stage.", throttle_duration_sec= 0.2)
        elif self.joy_stick_buffer.keys & self.WirelessButtons.L1:
            if self.run_stage == "policy":
                pass # nothing to do
            elif self.run_stage == "startup":
                self.run_stage = "policy"
                self.get_logger().info("Switched to policy stage.", throttle_duration_sec= 0.2)
                # set the motion reference refreshed time to the current time to prevent the policy from
                # being too laggy w.r.t motion reference target time.
                self.motion_ref_refreshed_time = self.get_clock().now()
            else:
                self.get_logger().warn(
                    "Should switch to startup stage before the policy runs. No switching. Current stage: " + self.run_stage,
                    throttle_duration_sec= 0.2,
                )

def load_onnx_sessions(model_dir: str) -> dict[str, ort.InferenceSession]:
    """ Load the ONNX model from the given directory.
    Args:
        model_dir: The directory which contains the ONNX model files.
    Returns:
        model: The dictionary of ONNX model with the key as the model name.
    """
    ort_execution_providers = ort.get_available_providers()
    # ort_execution_providers = ["CPUExecutionProvider"]
    print("Available ONNX execution providers: ", ort_execution_providers)
    sessions = dict()
    for model_name in os.listdir(model_dir):
        if not model_name.endswith(".onnx"):
            continue
        model_path = os.path.join(model_dir, model_name)
        sessions[model_name] = ort.InferenceSession(model_path, providers=ort_execution_providers)
    print("ONNX models loaded: ", sessions.keys())
    return sessions

def main(args):
    rclpy.init()

    # read params
    with open(os.path.join(args.logdir, "params", "env.yaml"), "r") as f:
        env_config = yaml.unsafe_load(f)
    with open(os.path.join(args.logdir, "params", "agent.yaml"), "r") as f:
        agent_config = yaml.unsafe_load(f)

    # load onnx model
    model_dir = os.path.join(args.logdir, "exported")
    onnx_sessions = load_onnx_sessions(model_dir)
    
    # start the node
    node = G1Node(
        agent_cfg=agent_config,
        cfg=env_config,
        interested_link_idx=args.link_idx,
        model_device=torch.device("mps") if args.runon == "mac" else torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        kp_factor=args.kp_factor,
        kd_factor=args.kd_factor,
        dryrun=not args.nodryrun,
    )
    node.register_network(onnx_sessions)

    # start the ros handlers and let the ROS2 spins
    node.start_ros_handlers()
    rclpy.spin(node)
    rclpy.shutdown()

    # # some testing code. do not run this when deploying the robot.
    if False:
        import time
        for loop_idx in range(1001):
            if loop_idx == 1: start_time = time.time()
            node._robot_forward_kinematics_callback()
            node.main_loop()
        end_time = time.time()
        print("Time elapsed: ", end_time - start_time)
        print("Average time: ", (end_time - start_time) / 1000)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--debug", action= "store_true", default= False, help= "Enable debug mode")
    parser.add_argument("--logdir", type= str, default= None, help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--link_idx", type= int, default= None, help= "The link index to viusalize the transfrom (computed by forward_kinematics ONNXProgram)")
    parser.add_argument("--kp_factor", type= float, default= 1.0, help= "The factor to scale the kp term")
    parser.add_argument("--kd_factor", type= float, default= 1.0, help= "The factor to scale the kd term")
    parser.add_argument("--runon", type= str, choices= ["mac", "thinkpad", "jetson", None], default= None, help= "The machine running the code")
    parser.add_argument("--nodryrun", action= "store_true", default= False, help= "Disable dryrun mode")

    args = parser.parse_args()

    if args.debug:
        import debugpy
        ip_address = ('0.0.0.0', 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args)