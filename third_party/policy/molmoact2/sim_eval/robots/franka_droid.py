import os
from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class FrankaDROID(BaseAgent):
    """Franka FR3 + Robotiq 2F-85 gripper with wrist and exterior DROID cameras."""

    uid = "franka_droid"

    urdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "assets", "franka_description", "urdfs", "fr3_robotiq.urdf",
    )

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            right_inner_finger_pad=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.0, -np.pi/5, 0, -np.pi*4/5, 0, np.pi*3/5, 0, 0, 0, 0, 0, 0.04, 0.04]),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
        "fr3_joint5", "fr3_joint6", "fr3_joint7",
    ]
    ee_link_name = "fr3_link8"

    arm_stiffness   = 1e3
    arm_damping     = 1e2
    arm_force_limit = 100

    gripper_stiffness   = 1e5
    gripper_damping     = 1e3
    gripper_force_limit = 0.1
    gripper_friction    = 0.05

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="wrist_cam",
                pose=sapien.Pose(p=[-0.074, 0.031, 0.011], q=[0.8038568606172174, 0, -0.5948227867513413, 0]),
                width=640, height=360,
                fov=2 * np.arctan(3.024 / (2 * 2.8)),  # DROID wrist cam: focal=2.8mm, v_ap=3.024mm
                near=0.01, far=100,
                mount=self.robot.links_map["robotiq_arg2f_base_link"],
            ),
            CameraConfig(
                uid="external_cam",
                pose=sapien_utils.look_at(eye=[0.05, 0.57, 0.66], target=[0.63, -0.48, -0.42]),
                width=640, height=360,
                fov=2 * np.arctan(3.024 / (2 * 2.1)),  # DROID exterior cam: focal=2.1mm, v_ap=3.024mm
                near=0.1, far=100,
                mount=self.robot.links_map["base"],
            ),
        ]

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=None, upper=None,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=-0.1, upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, ee_link=self.ee_link_name, urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1,
            rot_lower=-0.1, rot_upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, ee_link=self.ee_link_name, urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names, pos_lower=None, pos_upper=None,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, ee_link=self.ee_link_name, urdf_path=self.urdf_path,
            use_delta=False, normalize_action=False,
        )
        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names, -1.0, 1.0, self.arm_damping, self.arm_force_limit,
        )
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names, None, None,
            self.arm_stiffness, self.arm_damping, self.arm_force_limit, normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names, -0.1, 0.1,
            self.arm_stiffness, self.arm_damping, self.arm_force_limit, use_delta=True,
        )

        passive_finger_joints = PassiveControllerConfig(
            joint_names=[
                "left_inner_knuckle_joint", "right_inner_knuckle_joint",
                "left_inner_finger_joint",  "right_inner_finger_joint",
            ],
            damping=0, friction=0,
        )
        mimic_config = dict(
            left_outer_knuckle_joint=dict(joint="right_outer_knuckle_joint", multiplier=1.0, offset=0.0),
        )
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            ["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
            lower=None, upper=None,
            stiffness=self.gripper_stiffness, damping=self.gripper_damping,
            force_limit=self.gripper_force_limit, friction=self.gripper_friction,
            normalize_action=False, mimic=mimic_config,
        )
        g  = finger_mimic_pd_joint_pos
        gp = passive_finger_joints
        return deepcopy(dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=g, gripper_passive=gp),
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=g, gripper_passive=gp),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=g, gripper_passive=gp),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=g, gripper_passive=gp),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=g, gripper_passive=gp),
            pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos, gripper=g, gripper_passive=gp),
            pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos, gripper=g, gripper_passive=gp),
            pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose, gripper=g, gripper_passive=gp),
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=g, gripper_passive=gp),
            pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel, gripper=g, gripper_passive=gp),
            pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel, gripper=g, gripper_passive=gp),
        ))

    def _after_loading_articulation(self):
        outer_finger  = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # Magic poses from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [ 1.3578170e-09,-1.7901104e-02, 6.5159947e-03]
        p_f_left  = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left  = [-1.4041154e-08,-1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right))
        right_drive.set_limit_x(0, 0); right_drive.set_limit_y(0, 0); right_drive.set_limit_z(0, 0)

        outer_finger  = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()
        left_drive = self.scene.create_drive(lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left))
        left_drive.set_limit_x(0, 0); left_drive.set_limit_y(0, 0); left_drive.set_limit_z(0, 0)

        # SRDF would create too many collision groups; disable gripper self-collisions manually.
        for link_name in [
            "right_inner_knuckle", "right_outer_knuckle", "left_inner_knuckle", "left_outer_knuckle",
            "right_inner_finger_pad", "left_inner_finger_pad",
            "right_outer_finger", "left_outer_finger", "robotiq_arg2f_base_link",
            "right_inner_finger", "left_inner_finger", "fr3_link8",
        ]:
            self.robot.links_map[link_name].set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_inner_finger_pad")
        self.finger2_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_inner_finger_pad")
        self.tcp          = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        lf = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        rf = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        ld =  self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rd = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        lflag = (torch.linalg.norm(lf, axis=1) >= min_force) & (torch.rad2deg(common.compute_angle_between(ld, lf)) <= max_angle)
        rflag = (torch.linalg.norm(rf, axis=1) >= min_force) & (torch.rad2deg(common.compute_angle_between(rd, rf)) <= max_angle)
        return lflag & rflag

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
