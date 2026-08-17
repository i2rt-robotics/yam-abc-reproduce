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
class BimanualYAM(BaseAgent):
    """Bimanual YAM — two 6-DOF arms with linear grippers, loaded from a single MJCF."""

    uid = "bimanual_yam"

    mjcf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "assets", "yam", "yam_mujoco", "bimanual_yam_linear_flattened.xml",
    )
    urdf_path = None
    urdf_config = dict()

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0, -0.02, -0.02,
                0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0, -0.02, -0.02,
            ]),
            pose=sapien.Pose(),
        ),
        home=Keyframe(
            qpos=np.zeros(16),
            pose=sapien.Pose(),
        ),
    )

    left_arm_joint_names = [
        "left_joint1", "left_joint2", "left_joint3",
        "left_joint4", "left_joint5", "left_joint6",
    ]
    right_arm_joint_names = [
        "right_joint1", "right_joint2", "right_joint3",
        "right_joint4", "right_joint5", "right_joint6",
    ]
    arm_joint_names = left_arm_joint_names + right_arm_joint_names

    left_gripper_joint_names  = ["left_left_finger",  "left_right_finger"]
    right_gripper_joint_names = ["right_left_finger", "right_right_finger"]
    gripper_joint_names = left_gripper_joint_names + right_gripper_joint_names

    left_ee_link_name  = "left_link_6"
    right_ee_link_name = "right_link_6"

    # Gains matched to hardware MJCF actuators (dm4340 j1-3, override j4, dm4310 j5-6).
    # balance_passive_force=True matches MJCF gravcomp=1 so low gains don't cause sag.
    arm_stiffness   = [40.,  40.,  40.,  20.,  10.,  10.]
    arm_damping     = [2.5,  2.5,  2.5,  0.5,  1.0,  1.0]
    arm_force_limit = [28.,  28.,  28.,  10.,  10.,  10.]

    # kp high enough to saturate force_limit on contact → force_limit sets grip force.
    gripper_stiffness   = 2e3
    gripper_damping     = 40.
    gripper_force_limit = 40.

    # MJCF has no per-geom friction; override after load so fingers don't slip.
    gripper_static_friction  = 3.0
    gripper_dynamic_friction = 2.5

    # 360×640, matching MolmoAct2 YAM training images.
    cam_width  = 640
    cam_height = 360
    # Pinhole FOV from hardware: top = RealSense D435i (69.4°), wrists = D405 (87°).
    top_cam_hfov_deg   = 69.4
    wrist_cam_hfov_deg = 87.0

    @staticmethod
    def _intrinsic_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return np.array(
            [[fx, 0.0, width / 2.0],
             [0.0, fx, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    @property
    def _sensor_configs(self):
        w, h = self.cam_width, self.cam_height
        common_cfg = dict(width=w, height=h, near=0.01, far=100)
        top_K   = self._intrinsic_from_hfov(w, h, self.top_cam_hfov_deg)
        wrist_K = self._intrinsic_from_hfov(w, h, self.wrist_cam_hfov_deg)
        return [
            CameraConfig(
                uid="top_cam",
                pose=sapien.Pose(p=[0.15, 0, 0.8], q=[0.7660444431189782, 0, 0.6427876096865391, 0]),
                intrinsic=top_K,
                mount=self.robot.links_map["bimanual_base"],
                **common_cfg,
            ),
            CameraConfig(
                uid="left_cam",
                pose=sapien.Pose(p=[0, 0.09, 0.06], q=[0.612372429196013, -0.35355339154618404, -0.3535533966987049, -0.612372438120441]),
                intrinsic=wrist_K,
                mount=self.robot.links_map["left_link_6"],
                **common_cfg,
            ),
            CameraConfig(
                uid="right_cam",
                pose=sapien.Pose(p=[0, 0.09, 0.06], q=[0.612372429196013, -0.35355339154618404, -0.3535533966987049, -0.612372438120441]),
                intrinsic=wrist_K,
                mount=self.robot.links_map["right_link_6"],
                **common_cfg,
            ),
        ]

    @property
    def _controller_configs(self):
        def arm_cfg(joint_names):
            return PDJointPosControllerConfig(
                joint_names, lower=None, upper=None,
                stiffness=self.arm_stiffness, damping=self.arm_damping,
                force_limit=self.arm_force_limit, normalize_action=False,
            )

        def gripper_cfg(joint_names, mimic_key, mimic_target):
            return PDJointPosMimicControllerConfig(
                joint_names, lower=-0.0475, upper=0.0,
                stiffness=self.gripper_stiffness, damping=self.gripper_damping,
                force_limit=self.gripper_force_limit,
                mimic={mimic_key: {"joint": mimic_target}},
            )

        return deepcopy(dict(
            pd_joint_pos=dict(
                left_arm=arm_cfg(self.left_arm_joint_names),
                left_gripper=gripper_cfg(self.left_gripper_joint_names, "left_right_finger", "left_left_finger"),
                right_arm=arm_cfg(self.right_arm_joint_names),
                right_gripper=gripper_cfg(self.right_gripper_joint_names, "right_right_finger", "right_left_finger"),
            ),
        ))

    def _after_init(self):
        self.left_finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_link_left_finger")
        self.left_finger2_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_link_right_finger")
        self.left_tcp          = sapien_utils.get_obj_by_name(self.robot.get_links(), self.left_ee_link_name)
        self.right_finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_link_left_finger")
        self.right_finger2_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_link_right_finger")
        self.right_tcp          = sapien_utils.get_obj_by_name(self.robot.get_links(), self.right_ee_link_name)
        self._set_gripper_friction(self.gripper_static_friction, self.gripper_dynamic_friction)

    def _set_gripper_friction(self, static_friction: float, dynamic_friction: float):
        import sapien.physx as physx
        for link in [self.left_finger1_link, self.left_finger2_link,
                     self.right_finger1_link, self.right_finger2_link]:
            for body in link._bodies:
                for cs in body.get_collision_shapes():
                    cs.physical_material = physx.PhysxMaterial(
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                        restitution=0.0,
                    )

    def is_grasping(self, object: Actor, arm: str = "left", min_force=0.5, max_angle=85):
        if arm == "left":
            f1, f2 = self.left_finger1_link, self.left_finger2_link
        else:
            f1, f2 = self.right_finger1_link, self.right_finger2_link
        lf = self.scene.get_pairwise_contact_forces(f1, object)
        rf = self.scene.get_pairwise_contact_forces(f2, object)
        ld = f1.pose.to_transformation_matrix()[..., :3, 2]
        rd = -f2.pose.to_transformation_matrix()[..., :3, 2]
        lflag = (torch.linalg.norm(lf, axis=1) >= min_force) & (torch.rad2deg(common.compute_angle_between(ld, lf)) <= max_angle)
        rflag = (torch.linalg.norm(rf, axis=1) >= min_force) & (torch.rad2deg(common.compute_angle_between(rd, rf)) <= max_angle)
        return lflag & rflag

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-4]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def left_tcp_pos(self):
        return self.left_tcp.pose.p

    @property
    def right_tcp_pos(self):
        return self.right_tcp.pose.p

    @property
    def left_tcp_pose(self):
        return self.left_tcp.pose

    @property
    def right_tcp_pose(self):
        return self.right_tcp.pose
