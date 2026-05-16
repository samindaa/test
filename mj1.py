"""MuJoCo-based physics engine for MimicKit."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable

import engines.engine as engine
import engines.mjlab_recorder as mjlab_recorder
from mjlab.actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.mujoco import dof_width, qpos_width
import mujoco
import numpy as np
import torch


def _mj_to_mk_quat(q: torch.Tensor) -> torch.Tensor:
  """Convert MuJoCo (w,x,y,z) quaternion to MimicKit (x,y,z,w)."""
  return q[..., [1, 2, 3, 0]]


def _mk_to_mj_quat(q: torch.Tensor) -> torch.Tensor:
  """Convert MimicKit (x,y,z,w) quaternion to MuJoCo (w,x,y,z)."""
  return q[..., [3, 0, 1, 2]]


@dataclasses.dataclass(eq=False)
class ObjInfo:
  """Struct to store per-object indexing and metadata."""

  body_start: int = 0
  body_end: int = 0
  num_bodies: int = 0
  body_names: list[str] = dataclasses.field(default_factory=list)
  q_dof_start: int = 0
  q_dof_end: int = 0
  v_dof_start: int = 0
  v_dof_end: int = 0
  num_dofs: int = 0
  ctrl_start: int = 0
  ctrl_end: int = 0
  has_free_joint: bool = False
  has_sphere_joints: bool = False
  enable_self_collisions: bool = False
  sphere_q_starts: list[int] = dataclasses.field(default_factory=list)
  sphere_v_starts: list[int] = dataclasses.field(default_factory=list)
  nonsphere_q_indices: list[int] = dataclasses.field(default_factory=list)
  nonsphere_v_indices: list[int] = dataclasses.field(default_factory=list)
  asset_file: str = ""
  q_root_start: int = 0
  v_root_start: int = 0
  fix_root: bool = False
  disable_motors: bool = False
  color: list[float] | None = None
  obj_type: str = ""
  name: str = ""
  is_visual: bool = False
  start_pos: np.ndarray | None = None
  start_rot: np.ndarray | None = None
  per_joint_kp: list[float] = dataclasses.field(default_factory=list)
  per_joint_kd: list[float] = dataclasses.field(default_factory=list)

  def _config_key(self) -> tuple:
    """Return a hashable key from configuration fields."""
    color_key = tuple(self.color) if self.color is not None else None
    return (
        self.asset_file,
        self.fix_root,
        self.is_visual,
        self.enable_self_collisions,
        self.disable_motors,
        color_key,
    )

  def __hash__(self) -> int:
    return hash(self._config_key())

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, ObjInfo):
      return NotImplemented
    return self._config_key() == other._config_key()


class MjlabEngine(engine.Engine):
  """MuJoCo-based physics engine for MimicKit.

  Uses mjlab's Scene and Simulation for environment management,
  following the ManagerBasedRlEnv pattern. For ``pos`` control mode,
  uses mjlab's IdealPdActuator to convert position targets into torques.
  """

  def __init__(
      self,
      config: dict[str, Any],
      num_envs: int,
      device: str,
      visualize: bool,
      record_video: bool = False,
  ) -> None:
    super().__init__(visualize=visualize)

    self._vis: bool = visualize
    self._device: str = device
    self._num_envs: int = num_envs
    self._env_spacing: float = config.get("env_spacing", 5.0)

    sim_freq: int = config.get("sim_freq", 240)
    control_freq: int = config.get("control_freq", 30)
    assert (
        sim_freq >= control_freq and sim_freq % control_freq == 0
    ), "Simulation frequency must be a multiple of the control frequency"

    self._timestep: float = 1.0 / control_freq
    self._sim_steps: int = int(sim_freq / control_freq)
    self._sim_timestep: float = 1.0 / sim_freq
    self._sim_step_count: int = 0

    if "control_mode" in config:
      self._control_mode: engine.ControlMode = engine.ControlMode[
          config["control_mode"]
      ]
    else:
      self._control_mode = engine.ControlMode.none

    # Solver settings
    self._solver_type: str = config.get("solver", "newton")
    self._integrator: str = config.get("integrator", "implicitfast")
    self._solver_iterations: int = config.get("iterations", 100)
    self._ls_iterations: int = config.get("ls_iterations", 50)

    # Object tracking
    self._env_obj_infos: list[list[ObjInfo]] = []
    self._obj_info_cache: dict[int, ObjInfo] = {}
    self._spec_cache: dict[str, mujoco.MjSpec] = {}

    if visualize:
      self._record_video: bool = False
    else:
      self._record_video = record_video

    # Camera state
    self._cam_pos: np.ndarray = np.array([0.0, -5.0, 3.0])
    self._cam_dir: np.ndarray = np.array([0.0, 1.0, -0.3])

    # Viewer
    self._viewer: Any | None = None

  def get_name(self) -> str:
    return "mjlab"

  def create_env(self) -> int:
    env_id = len(self._env_obj_infos)
    assert env_id < self.get_num_envs()
    self._env_obj_infos.append([])
    return env_id

  def create_obj(
      self,
      env_id: int,
      obj_type: str,
      asset_file: str,
      name: str,
      is_visual: bool = False,
      enable_self_collisions: bool = True,
      fix_root: bool = False,
      start_pos: np.ndarray | None = None,
      start_rot: np.ndarray | None = None,
      color: list[float] | None = None,
      disable_motors: bool = False,
  ) -> int:
    if start_rot is None:
      start_rot = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw identity

    if start_pos is None:
      start_pos = np.array([0.0, 0.0, 0.0])

    obj_info = ObjInfo(
        obj_type=obj_type,
        asset_file=asset_file,
        name=name,
        fix_root=fix_root,
        is_visual=is_visual,
        enable_self_collisions=enable_self_collisions,
        start_pos=start_pos,
        start_rot=start_rot,
        color=color,
        disable_motors=disable_motors,
    )

    # Reuse existing ObjInfo for identical configs
    # (like newton's _builder_cache)
    cache_key = hash(obj_info)
    if cache_key in self._obj_info_cache:
      cached = self._obj_info_cache[cache_key]
      if cached == obj_info:
        obj_info = cached
    else:
      self._obj_info_cache[cache_key] = obj_info

    obj_id = len(self._env_obj_infos[env_id])
    self._env_obj_infos[env_id].append(obj_info)
    return obj_id

  def initialize_sim(self) -> None:
    self._validate_envs()

    logging.info("Building Scene (ManagerBasedRlEnv pattern)...")
    self._obj_infos: list[ObjInfo] = list(self._env_obj_infos[0])

    # Build EntityCfg for each object
    entities: dict[str, EntityCfg] = {}
    for obj_idx, obj_info in enumerate(self._obj_infos):
      entity_name = f"obj{obj_idx}"
      entities[entity_name] = self._make_entity_cfg(obj_info)

    # Build SceneCfg
    scene_cfg = SceneCfg(
        num_envs=self._num_envs,
        env_spacing=self._env_spacing,
        entities=entities,
        terrain=TerrainEntityCfg(terrain_type="plane"),
    )

    # Create Scene → compile → Simulation (ManagerBasedRlEnv pattern)
    self._scene = Scene(scene_cfg, device=self._device)

    logging.info("Creating Simulation...")
    sim_cfg = SimulationCfg(
        njmax=500,
        nconmax=300,
        mujoco=MujocoCfg(
            timestep=self._sim_timestep,
            solver=self._solver_type,
            integrator=self._integrator,
            iterations=self._solver_iterations,
            ls_iterations=self._ls_iterations,
        ),
    )

    self._sim = Simulation(
        num_envs=self._num_envs,
        cfg=sim_cfg,
        model=self._scene.compile(),
        device=self._device,
    )
    self._mj_model = self._sim.mj_model

    # Initialize scene (wires entities to model/data)
    self._scene.initialize(
        mj_model=self._mj_model,
        model=self._sim.model,
        data=self._sim.data,
    )
    for obj_idx in range(len(self._obj_infos)):
      obj = self._obj_infos[obj_idx]
      entity = self._scene.entities[f"obj{obj_idx}"]

      # Populate basic indexing from Entity
      obj.body_start = (
          entity.indexing.body_ids[0]
          if len(entity.indexing.body_ids) > 0
          else 0
      )
      obj.body_end = (
          entity.indexing.body_ids[-1] + 1
          if len(entity.indexing.body_ids) > 0
          else 0
      )
      obj.num_bodies = entity.num_bodies
      obj.body_names = list(entity.body_names)

      obj.q_dof_start = (
          entity.indexing.joint_q_adr[0]
          if len(entity.indexing.joint_q_adr) > 0
          else 0
      )
      obj.q_dof_end = (
          entity.indexing.joint_q_adr[-1] + 1
          if len(entity.indexing.joint_q_adr) > 0
          else 0
      )
      obj.v_dof_start = (
          entity.indexing.joint_v_adr[0]
          if len(entity.indexing.joint_v_adr) > 0
          else 0
      )
      obj.v_dof_end = (
          entity.indexing.joint_v_adr[-1] + 1
          if len(entity.indexing.joint_v_adr) > 0
          else 0
      )
      obj.num_dofs = len(entity.indexing.joint_v_adr)

      obj.ctrl_start = (
          entity.indexing.ctrl_ids[0]
          if len(entity.indexing.ctrl_ids) > 0
          else 0
      )
      obj.ctrl_end = (
          entity.indexing.ctrl_ids[-1] + 1
          if len(entity.indexing.ctrl_ids) > 0
          else 0
      )

      obj.has_free_joint = not entity.is_fixed_base
      if not entity.is_fixed_base:
        obj.q_root_start = entity.indexing.free_joint_q_adr[0]
        obj.v_root_start = entity.indexing.free_joint_v_adr[0]

      # Sphere joint computation
      has_sphere = False
      sphere_q = []
      sphere_v = []
      nonsphere_q = []
      nonsphere_v = []

      for name in entity.joint_names:
        jnt = self._mj_model.joint(f"obj{obj_idx}/{name}")
        jnt_type = jnt.type[0]
        qadr = jnt.qposadr[0]
        vadr = jnt.dofadr[0]
        if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
          has_sphere = True
          sphere_q.append(qadr)
          sphere_v.append(vadr)
        else:
          nonsphere_q.extend(range(qadr, qadr + qpos_width(jnt_type)))
          nonsphere_v.extend(range(vadr, vadr + dof_width(jnt_type)))

      obj.has_sphere_joints = has_sphere
      obj.sphere_q_starts = sphere_q
      obj.sphere_v_starts = sphere_v
      obj.nonsphere_q_indices = nonsphere_q
      obj.nonsphere_v_indices = nonsphere_v

    # Set per-joint PD gains on IdealPdActuator (overrides the scalar default)
    if self._control_mode == engine.ControlMode.pos:
      for obj_idx, obj in enumerate(self._obj_infos):
        if not obj.per_joint_kp:
          continue
        entity = self._scene.entities[f"obj{obj_idx}"]
        for act in entity._custom_actuators:  # pylint: disable=protected-access
          if isinstance(act, IdealPdActuator):
            num_targets = len(act.target_names)
            kp_arr = obj.per_joint_kp
            kd_arr = obj.per_joint_kd
            # If gains list doesn't match actuator target count (e.g. free
            # joint was excluded), trim or pad with last value
            if len(kp_arr) > num_targets:
              kp_arr = kp_arr[:num_targets]
              kd_arr = kd_arr[:num_targets]
            elif len(kp_arr) < num_targets:
              pad_kp = kp_arr[-1] if kp_arr else 0.0
              pad_kd = kd_arr[-1] if kd_arr else 0.0
              kp_arr = kp_arr + [pad_kp] * (num_targets - len(kp_arr))
              kd_arr = kd_arr + [pad_kd] * (num_targets - len(kd_arr))
            kp_t = torch.tensor(
                [kp_arr], device=self._device, dtype=torch.float32
            ).expand(self._num_envs, -1)
            kd_t = torch.tensor(
                [kd_arr], device=self._device, dtype=torch.float32
            ).expand(self._num_envs, -1)
            act.set_gains(slice(None), kp=kp_t, kd=kd_t)  # pytype: disable=attribute-error
            logging.info(
                "Set per-joint PD gains on obj%d: kp=%s, kd=%s",
                obj_idx, kp_arr, kd_arr,
            )
            break

    # Create a display MjData for the viewer
    self._mj_data_display = mujoco.MjData(self._mj_model)
    mujoco.mj_forward(self._mj_model, self._mj_data_display)

    self._build_controls()
    self._build_contact_buffers()
    self._apply_start_xform()

    if self._visualize():
      self._build_viewer()

    if self.enabled_record_video():
      self._video_recorder = self._build_video_recorder()
      self._recording = False

    logging.info(
        "MuJoCo model: nbody={}, njnt={}, nq={}, nv={}, nu={}".format(
            self._mj_model.nbody,
            self._mj_model.njnt,
            self._mj_model.nq,
            self._mj_model.nv,
            self._mj_model.nu,
        )
    )

  def _make_entity_cfg(self, obj_info: ObjInfo) -> EntityCfg:
    """Create an EntityCfg from an ObjInfo, with optional IdealPdActuator."""
    asset_file = obj_info.asset_file

    if "g1.xml" in asset_file:
      joint_pos = {
          ".*_hip_pitch_joint": -0.1,
          ".*_knee_joint": 0.3,
          ".*_ankle_pitch_joint": -0.2,
          ".*_shoulder_pitch_joint": 0.2,
          ".*_elbow_joint": 1.28,
          "left_shoulder_roll_joint": 0.2,
          "right_shoulder_roll_joint": -0.2,
      }
    else:
      # Default to zero joint positions. Models with keyframes could use None
      # to read from the keyframe, but not all models have keyframes.
      joint_pos = {".*": 0.0}

    def make_spec_fn(f: str, info: ObjInfo) -> Callable[[], mujoco.MjSpec]:
      def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(f)
        if info.fix_root:
          weld = spec.add_equality(
              name="weld_pelvis_to_space",
              type=mujoco.mjtEq.mjEQ_WELD,
              objtype=mujoco.mjtObj.mjOBJ_BODY,
              name1="pelvis",
          )
          weld.data[:7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
          logging.info("Added weld constraint for pelvis to fix root.")
        if info.disable_motors:
          for actuator in spec.actuators:
            actuator.gainprm[0] = 0.0
            actuator.biasprm[:] = 0.0
        if info.color is not None:
          col_rgba = (
              np.array([*info.color, 1.0])
              if len(info.color) == 3
              else np.array(info.color)
          )
          for geom in spec.geoms:
            geom.rgba = col_rgba
        return spec

      return spec_fn

    # Build articulation config for actuated objects
    if self._control_mode == engine.ControlMode.pos:
      tmp_spec = mujoco.MjSpec.from_file(asset_file)
      # Extract per-joint gains from the MJCF model
      per_joint_kp = []
      per_joint_kd = []
      effort_limit = float("inf")
      for joint in tmp_spec.joints:
        if not hasattr(joint, "stiffness"):
          continue
        stiff = np.asarray(joint.stiffness).flatten()
        damp = np.asarray(joint.damping).flatten()
        jkp = float(stiff[0]) if stiff.size > 0 else 0.0
        jkd = float(damp[0]) if damp.size > 0 else 0.0
        if jkp > 0:
          per_joint_kp.append(jkp)
          per_joint_kd.append(jkd)
          # Use the first valid effort limit found
          if effort_limit == float("inf"):
            if hasattr(joint, "actuatorfrcrange"):
              frc = np.asarray(joint.actuatorfrcrange).flatten()
              if frc.size > 1 and frc[1] > 0:
                effort_limit = float(frc[1])
            elif hasattr(joint, "actfrcrange"):
              frc = np.asarray(joint.actfrcrange).flatten()
              if frc.size > 1 and frc[1] > 0:
                effort_limit = float(frc[1])

      # Use mean as the initial scalar for IdealPdActuatorCfg (will be
      # overridden per-joint after scene.initialize)
      kp = float(np.mean(per_joint_kp)) if per_joint_kp else 0.0
      kd = float(np.mean(per_joint_kd)) if per_joint_kd else 0.0
      obj_info.per_joint_kp = per_joint_kp
      obj_info.per_joint_kd = per_joint_kd
      print(
          f"Extracted per-joint gains: kp={per_joint_kp},"
          f" kd={per_joint_kd}, effort_limit={effort_limit}"
      )

      # Remove the existing actuators — IdealPdActuator will add its own
      # motor actuators
      def make_spec_fn_no_actuators(
          f: str, info: ObjInfo
      ) -> Callable[[], mujoco.MjSpec]:
        def spec_fn() -> mujoco.MjSpec:
          spec = mujoco.MjSpec.from_file(f)
          if info.fix_root:
            weld = spec.add_equality(
                name="weld_pelvis_to_space",
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1="pelvis",
            )
            weld.active = True
            weld.data[:7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            logging.info("Added weld constraint for pelvis to fix root.")
          # Remove existing actuators so IdealPdActuator can add clean ones
          while spec.actuators:
            spec.delete(spec.actuators[0])
          if info.color is not None:
            col_rgba = (
                np.array([*info.color, 1.0])
                if len(info.color) == 3
                else np.array(info.color)
            )
            for geom in spec.geoms:
              geom.rgba = col_rgba
          return spec

        return spec_fn

      articulation = EntityArticulationInfoCfg(
          actuators=(
              IdealPdActuatorCfg(
                  target_names_expr=(".*",),
                  stiffness=kp,
                  damping=kd,
                  effort_limit=effort_limit,
              ),
          ),
      )
      return EntityCfg(
          spec_fn=make_spec_fn_no_actuators(asset_file, obj_info),
          articulation=articulation,
          init_state=EntityCfg.InitialStateCfg(
              pos=obj_info.start_pos,
              rot=obj_info.start_rot,
              joint_pos=joint_pos,
          ),
      )

    return EntityCfg(
        spec_fn=make_spec_fn(asset_file, obj_info),
        init_state=EntityCfg.InitialStateCfg(
            pos=obj_info.start_pos,
            rot=obj_info.start_rot,
            joint_pos=joint_pos,
        ),
    )

  def step(self) -> None:
    if self.enabled_record_video() and self._recording:
      self._video_recorder.capture_frame()

    self._simulate()
    self._update_contact_buffers()
    self._sim_step_count += 1

  def reset_envs(self, env_ids: torch.Tensor) -> None:
    """Reset specified environments."""
    self._sim.reset(env_ids)
    self._scene.reset(env_ids)
    # Re-apply initial transforms for the reset envs
    for obj_idx, obj in enumerate(self._obj_infos):
      start_pos = torch.tensor(
          obj.start_pos, device=self._device, dtype=torch.float32
      )
      start_rot_xyzw = torch.tensor(
          obj.start_rot, device=self._device, dtype=torch.float32
      )
      self.set_root_pos(env_ids, obj_idx, start_pos)
      self.set_root_rot(env_ids, obj_idx, start_rot_xyzw)
    self._scene.write_data_to_sim()
    self._sim.forward()
    self._sim.sense()

  def set_cmd(self, obj_id: int, cmd: torch.Tensor) -> None:
    obj = self._obj_infos[obj_id]
    ctrl_slice = slice(obj.ctrl_start, obj.ctrl_end)

    if self._control_mode == engine.ControlMode.none:
      pass
    elif self._control_mode == engine.ControlMode.pos:
      # IdealPdActuator: set position target on the entity
      entity = self._scene.entities[f"obj{obj_id}"]
      entity.set_joint_position_target(cmd)
    elif self._control_mode == engine.ControlMode.vel:
      self._sim.data.ctrl[:, ctrl_slice] = cmd
    elif self._control_mode == engine.ControlMode.torque:
      self._sim.data.ctrl[:, ctrl_slice] = cmd
    elif self._control_mode == engine.ControlMode.pd_explicit:
      self._pd_targets[obj_id][:] = cmd
    else:
      assert False, "Unsupported control mode: {}".format(self._control_mode)

  def set_camera_pose(
      self, pos: np.ndarray | list[float], look_at: np.ndarray | list[float]
  ) -> None:
    self._cam_pos = np.array(pos, dtype=np.float64)
    cam_dir = np.array(look_at) - np.array(pos)
    norm = np.linalg.norm(cam_dir)
    if norm > 1e-8:
      cam_dir = cam_dir / norm
    self._cam_dir = cam_dir

    if self._viewer is not None:
      self._viewer.cam.lookat[:] = look_at
      self._viewer.cam.distance = np.linalg.norm(
          np.array(pos) - np.array(look_at)
      )

      dx = look_at[0] - pos[0]
      dy = look_at[1] - pos[1]
      dz = look_at[2] - pos[2]
      horiz = np.sqrt(dx * dx + dy * dy)
      self._viewer.cam.elevation = (
          np.rad2deg(np.arctan2(dz, horiz)) if horiz > 1e-8 else 0
      )
      self._viewer.cam.azimuth = np.rad2deg(np.arctan2(dy, dx))

  def get_camera_pos(self) -> np.ndarray:
    return self._cam_pos.copy()

  def get_camera_dir(self) -> np.ndarray:
    return self._cam_dir.copy()

  def render(self) -> None:
    if self._viewer is not None:
      # Sync env 0 state to the display MjData
      self._sync_to_mj_single(0, self._mj_data_display)
      mujoco.mj_forward(self._mj_model, self._mj_data_display)
      self._viewer.sync()
    super().render()  # pytype: disable=attribute-error

  def get_timestep(self) -> float:
    return self._timestep

  def get_sim_timestep(self) -> float:
    return self._sim_timestep

  def get_sim_time(self) -> float:
    return self._timestep * self._sim_step_count

  def get_num_envs(self) -> int:
    return self._num_envs

  def get_gravity(self) -> np.ndarray:
    return np.array(self._mj_model.opt.gravity)

  # ==================== State getters ====================

  def get_root_pos(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    if obj.has_free_joint:
      return self._sim.data.qpos[:, obj.q_root_start : obj.q_root_start + 3]
    else:
      return self._sim.data.xpos[:, obj.body_start, :]

  def get_root_rot(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    if obj.has_free_joint:
      rot_wxyz = self._sim.data.qpos[
          :, obj.q_root_start + 3 : obj.q_root_start + 7
      ]
    else:
      rot_wxyz = self._sim.data.xquat[:, obj.body_start, :]
    return _mj_to_mk_quat(rot_wxyz)

  def get_root_vel(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    if obj.has_free_joint:
      return self._sim.data.qvel[:, obj.v_root_start : obj.v_root_start + 3]
    else:
      return self._sim.data.cvel[:, obj.body_start, 3:6]

  def get_root_ang_vel(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    if obj.has_free_joint:
      ang_vel_body = self._sim.data.qvel[
          :, obj.v_root_start + 3 : obj.v_root_start + 6
      ]
      root_quat_wxyz = self._sim.data.qpos[
          :, obj.q_root_start + 3 : obj.q_root_start + 7
      ]
      return quat_apply(root_quat_wxyz, ang_vel_body)
    return self._sim.data.cvel[:, obj.body_start, 0:3]

  def get_dof_pos(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    if obj.has_sphere_joints:
      return self._get_dof_pos_expmap(obj)
    return self._sim.data.qpos[:, obj.q_dof_start : obj.q_dof_end]

  def get_dof_vel(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    return self._sim.data.qvel[:, obj.v_dof_start : obj.v_dof_end]

  def get_dof_forces(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    return self._sim.data.actuator_force[:, obj.ctrl_start : obj.ctrl_end]

  def get_body_pos(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    return self._sim.data.xpos[:, obj.body_start : obj.body_end, :]

  def get_body_rot(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    rot_wxyz = self._sim.data.xquat[:, obj.body_start : obj.body_end, :]
    return _mj_to_mk_quat(rot_wxyz)

  def get_body_vel(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    return self._sim.data.cvel[:, obj.body_start : obj.body_end, 3:6]

  def get_body_ang_vel(self, obj_id: int) -> torch.Tensor:
    obj = self._obj_infos[obj_id]
    return self._sim.data.cvel[:, obj.body_start : obj.body_end, 0:3]

  def get_contact_forces(self, obj_id: int) -> torch.Tensor:
    return self._contact_forces[obj_id]

  def get_ground_contact_forces(self, obj_id: int) -> torch.Tensor:
    return self._ground_contact_forces[obj_id]

  # ==================== State setters ====================

  def set_root_pos(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      root_pos: torch.Tensor,
  ) -> None:
    obj = self._obj_infos[obj_id]
    if obj.has_free_joint:
      q_start = obj.q_root_start
      if env_id is None:
        self._sim.data.qpos[:, q_start : q_start + 3] = root_pos
      else:
        self._sim.data.qpos[env_id, q_start : q_start + 3] = root_pos
    else:
      logging.warning(
          "set_root_pos on fixed-root obj %s writes to xpos "
          "which is overwritten by forward(). Use free joints to move objects.",
          obj_id,
      )
      if env_id is None:
        self._sim.data.xpos[:, obj.body_start, :] = root_pos
      else:
        self._sim.data.xpos[env_id, obj.body_start, :] = root_pos

  def set_root_rot(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      root_rot: torch.Tensor,
  ) -> None:
    """root_rot in MimicKit convention (x,y,z,w)."""
    obj = self._obj_infos[obj_id]
    rot_wxyz = _mk_to_mj_quat(root_rot)
    if obj.has_free_joint:
      q_start = obj.q_root_start + 3
      if env_id is None:
        self._sim.data.qpos[:, q_start : q_start + 4] = rot_wxyz
      else:
        self._sim.data.qpos[env_id, q_start : q_start + 4] = rot_wxyz
    else:
      logging.warning(
          "set_root_rot on fixed-root obj %s writes to xquat "
          "which is overwritten by forward(). Use free joints to move objects.",
          obj_id,
      )
      if env_id is None:
        self._sim.data.xquat[:, obj.body_start, :] = rot_wxyz
      else:
        self._sim.data.xquat[env_id, obj.body_start, :] = rot_wxyz

  def set_root_vel(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      root_vel: torch.Tensor,
  ) -> None:
    obj = self._obj_infos[obj_id]
    if obj.has_free_joint:
      v_start = obj.v_root_start
      if env_id is None:
        self._sim.data.qvel[:, v_start : v_start + 3] = root_vel
      else:
        self._sim.data.qvel[env_id, v_start : v_start + 3] = root_vel

  def set_root_ang_vel(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      root_ang_vel: torch.Tensor | int | float,
  ) -> None:
    """root_ang_vel is in world frame."""
    obj = self._obj_infos[obj_id]
    if obj.has_free_joint:
      v_start = obj.v_root_start + 3
      if isinstance(root_ang_vel, (int, float)):
        if env_id is None:
          self._sim.data.qvel[:, v_start : v_start + 3] = root_ang_vel
        else:
          self._sim.data.qvel[env_id, v_start : v_start + 3] = root_ang_vel
      else:
        root_quat_wxyz = self._sim.data.qpos[
            :, obj.q_root_start + 3 : obj.q_root_start + 7
        ]
        if env_id is not None:
          root_quat_wxyz = root_quat_wxyz[env_id]
        ang_vel_body = quat_apply_inverse(root_quat_wxyz, root_ang_vel)
        if env_id is None:
          self._sim.data.qvel[:, v_start : v_start + 3] = ang_vel_body
        else:
          self._sim.data.qvel[env_id, v_start : v_start + 3] = ang_vel_body

  def set_dof_pos(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      dof_pos: torch.Tensor | int | float,
  ) -> None:
    obj = self._obj_infos[obj_id]
    if obj.has_sphere_joints:
      self._set_dof_pos_with_sphere(env_id, obj, dof_pos)
    else:
      if env_id is None:
        self._sim.data.qpos[:, obj.q_dof_start : obj.q_dof_end] = dof_pos
      else:
        self._sim.data.qpos[env_id, obj.q_dof_start : obj.q_dof_end] = dof_pos

  def set_dof_vel(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      dof_vel: torch.Tensor,
  ) -> None:
    obj = self._obj_infos[obj_id]
    if env_id is None:
      self._sim.data.qvel[:, obj.v_dof_start : obj.v_dof_end] = dof_vel
    else:
      self._sim.data.qvel[env_id, obj.v_dof_start : obj.v_dof_end] = dof_vel

  def set_body_vel(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      body_vel: torch.Tensor,
  ) -> None:
    pass

  def set_body_ang_vel(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      body_ang_vel: torch.Tensor,
  ) -> None:
    pass

  def set_body_pos(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      body_pos: torch.Tensor,
  ) -> None:
    obj = self._obj_infos[obj_id]
    if env_id is None:
      self._sim.data.xpos[:, obj.body_start : obj.body_end, :] = body_pos
    else:
      self._sim.data.xpos[env_id, obj.body_start : obj.body_end, :] = body_pos

  def set_body_rot(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      body_rot: torch.Tensor,
  ) -> None:
    """body_rot in MimicKit convention (x,y,z,w)."""
    obj = self._obj_infos[obj_id]
    rot_wxyz = _mk_to_mj_quat(body_rot)
    if env_id is None:
      self._sim.data.xquat[:, obj.body_start : obj.body_end, :] = rot_wxyz
    else:
      self._sim.data.xquat[env_id, obj.body_start : obj.body_end, :] = rot_wxyz

  def set_body_forces(
      self,
      env_id: int | torch.Tensor | None,
      obj_id: int,
      body_id: int,
      forces: torch.Tensor,
  ) -> None:
    obj = self._obj_infos[obj_id]
    global_body_id = obj.body_start + body_id
    if env_id is None:
      self._sim.data.xfrc_applied[:, global_body_id, :3] = forces
    else:
      self._sim.data.xfrc_applied[env_id, global_body_id, :3] = forces

  # ==================== Object info ====================

  def get_obj_type(self, obj_id: int) -> str:
    return self._obj_infos[obj_id].obj_type

  def get_obj_num_dofs(self, obj_id: int) -> int:
    return self._obj_infos[obj_id].num_dofs

  def get_obj_num_bodies(self, obj_id: int) -> int:
    return self._obj_infos[obj_id].num_bodies

  def get_obj_body_names(self, obj_id: int) -> list[str]:
    return self._obj_infos[obj_id].body_names

  def find_obj_body_id(self, obj_id: int, body_name: str) -> int:
    body_names = self.get_obj_body_names(obj_id)
    body_id = body_names.index(body_name)
    return body_id

  def get_obj_torque_limits(self, env_id: int, obj_id: int) -> np.ndarray:
    obj = self._obj_infos[obj_id]
    if obj.ctrl_end > obj.ctrl_start:
      force_range = self._mj_model.actuator_forcerange[
          obj.ctrl_start : obj.ctrl_end
      ]
      return force_range[:, 1]
    return np.array([])

  def get_obj_dof_limits(
      self, env_id: int, obj_id: int
  ) -> tuple[np.ndarray, np.ndarray]:
    entity = self._scene.entities[f"obj{obj_id}"]
    dof_low = []
    dof_high = []
    for name in entity.joint_names:
      jnt = self._mj_model.joint(f"obj{obj_id}/{name}")
      jnt_id = jnt.id
      jnt_type = jnt.type[0]
      if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        lim = self._mj_model.jnt_range[jnt_id]
        for d in range(3):
          dof_low.append(
              lim[0] if self._mj_model.jnt_limited[jnt_id] else -np.pi
          )
          dof_high.append(
              lim[1] if self._mj_model.jnt_limited[jnt_id] else np.pi
          )
      else:
        lim = self._mj_model.jnt_range[jnt_id]
        dof_low.append(
            lim[0] if self._mj_model.jnt_limited[jnt_id] else -2 * np.pi
        )
        dof_high.append(
            lim[1] if self._mj_model.jnt_limited[jnt_id] else 2 * np.pi
        )
    return np.array(dof_low, dtype=np.float32), np.array(
        dof_high, dtype=np.float32
    )

  def get_obj_pd_gains(
      self, env_id: int, obj_id: int
  ) -> tuple[np.ndarray, np.ndarray]:
    obj = self._obj_infos[obj_id]
    kp = []
    kd = []
    for a_id in range(obj.ctrl_start, obj.ctrl_end):
      kp.append(self._mj_model.actuator_gainprm[a_id, 0])
      kd.append(self._mj_model.actuator_biasprm[a_id, 2])
    kp = np.array(kp, dtype=np.float32)
    kd = np.abs(np.array(kd, dtype=np.float32))
    return kp, kd

  def calc_obj_mass(self, env_id: int, obj_id: int) -> float:
    obj = self._obj_infos[obj_id]
    masses = self._mj_model.body_mass[obj.body_start : obj.body_end]
    return float(np.sum(masses))

  def get_control_mode(self) -> engine.ControlMode:
    return self._control_mode

  def get_sim_model(self) -> mujoco.MjModel:
    return self._mj_model

  def get_sim_data(self) -> Any:
    return self._sim.data

  def get_sim_data_env(self, env_id: int) -> Any:
    return self._sim.data

  def draw_lines(
      self,
      env_id: int,
      start_verts: Any,
      end_verts: Any,
      cols: Any,
      line_width: float,
  ) -> None:
    pass

  def register_keyboard_callback(
      self, key_str: str, callback_func: Callable[..., Any]
  ) -> None:
    pass

  def enabled_record_video(self) -> bool:
    return self._record_video

  def get_video_recording(self) -> Any:
    return self._video_recorder.get_video()

  def start_video_recording(self) -> None:
    self._video_recorder.clear()
    self._recording = True

  def stop_video_recording(self) -> None:
    self._recording = False

  # ==================== Private methods ====================

  def _visualize(self) -> bool:
    return self._vis

  def _validate_envs(self) -> None:
    num_envs = self.get_num_envs()
    objs_per_env = len(self._env_obj_infos[0])
    for i in range(num_envs):
      assert (
          len(self._env_obj_infos[i]) == objs_per_env
      ), "All envs must have the same number of objects."

  def _build_contact_buffers(self) -> None:
    """Allocate contact force buffers per object."""
    self._contact_forces = []
    self._ground_contact_forces = []
    for obj_idx in range(len(self._obj_infos)):
      entity = self._scene.entities[f"obj{obj_idx}"]
      shape = (self._num_envs, entity.num_bodies, 3)
      self._contact_forces.append(
          torch.zeros(shape, device=self._device, dtype=torch.float32)
      )
      self._ground_contact_forces.append(
          torch.zeros(shape, device=self._device, dtype=torch.float32)
      )

  def _update_contact_buffers(self) -> None:
    """Update contact force buffers from MuJoCo contact data."""
    for i in range(len(self._obj_infos)):
      self._contact_forces[i].zero_()
      self._ground_contact_forces[i].zero_()

    cfrc_ext = self._sim.data.cfrc_ext
    for obj_idx in range(len(self._obj_infos)):
      entity = self._scene.entities[f"obj{obj_idx}"]
      obj_cfrc = cfrc_ext[:, entity.indexing.body_ids, :]
      # Forces are in columns 3:6 (linear part)
      self._contact_forces[obj_idx] = obj_cfrc[:, :, 3:6].clone()
      self._ground_contact_forces[obj_idx] = obj_cfrc[:, :, 3:6].clone()

  def _apply_start_xform(self) -> None:
    """Apply initial positions and rotations to all objects."""
    for obj_idx, obj in enumerate(self._obj_infos):
      start_pos = torch.tensor(
          obj.start_pos, device=self._device, dtype=torch.float32
      )
      start_rot_xyzw = torch.tensor(
          obj.start_rot, device=self._device, dtype=torch.float32
      )

      self.set_root_pos(None, obj_idx, start_pos)
      self.set_root_rot(None, obj_idx, start_rot_xyzw)

    # Run forward to update FK
    self._sim.forward()

  def _simulate(self) -> None:
    """Run simulation substeps following the ManagerBasedRlEnv pattern.

    Each substep: write_data_to_sim → step → update.
    For pos mode, Scene.write_data_to_sim() triggers
    Entity._apply_actuator_controls()
    which calls IdealPdActuator.compute() to produce PD torques from the current
    joint state and position targets. This must happen every substep so the PD
    controller sees updated joint positions/velocities.
    """
    for _ in range(self._sim_steps):
      # For pd_explicit mode, compute and write PD torques each substep
      if self._control_mode == engine.ControlMode.pd_explicit:
        self._apply_pd_explicit_torque()

      # Scene.write_data_to_sim() → Entity.write_data_to_sim()
      # → Entity._apply_actuator_controls() which:
      #   - For IdealPdActuator (pos mode): reads joint_pos_target from
      #     entity.data, reads current joint_pos/joint_vel from sim.data,
      #     computes PD torque, writes to sim.data.ctrl
      #   - For builtin actuators: passes through targets to ctrl
      self._scene.write_data_to_sim()
      self._sim.step()
      # Scene.update() → Entity.update() → Actuator.update()
      self._scene.update(dt=self._sim_timestep)

  def _apply_pd_explicit_torque(self) -> None:
    """Compute and apply PD torques for pd_explicit mode."""
    for obj_idx, obj in enumerate(self._obj_infos):
      if obj.ctrl_end <= obj.ctrl_start:
        continue

      dof_pos = self.get_dof_pos(obj_idx)
      dof_vel = self.get_dof_vel(obj_idx)
      target = self._pd_targets[obj_idx]
      kp = self._pd_kp[obj_idx]
      kd = self._pd_kd[obj_idx]
      torque_lim = self._pd_torque_lim[obj_idx]

      torque = kp * (target - dof_pos) - kd * dof_vel
      torque = torch.clamp(torque, -torque_lim, torque_lim)

      ctrl_slice = slice(obj.ctrl_start, obj.ctrl_end)
      self._sim.data.ctrl[:, ctrl_slice] = torque

  def _build_controls(self) -> None:
    """Set up control mode buffers for pd_explicit."""
    if self._control_mode == engine.ControlMode.pd_explicit:
      self._pd_targets = []
      self._pd_kp = []
      self._pd_kd = []
      self._pd_torque_lim = []
      for obj_idx, obj in enumerate(self._obj_infos):
        num_ctrl = obj.ctrl_end - obj.ctrl_start
        target = torch.zeros(
            self._num_envs, num_ctrl, device=self._device, dtype=torch.float32
        )
        self._pd_targets.append(target)

        kp, kd = self.get_obj_pd_gains(0, obj_idx)
        self._pd_kp.append(
            torch.tensor(kp, device=self._device, dtype=torch.float32)
        )
        self._pd_kd.append(
            torch.tensor(kd, device=self._device, dtype=torch.float32)
        )

        torque_lim = self.get_obj_torque_limits(0, obj_idx)
        self._pd_torque_lim.append(
            torch.tensor(torque_lim, device=self._device, dtype=torch.float32)
        )

  def _sync_to_mj_single(self, env_idx: int, mj_data: mujoco.MjData) -> None:
    """Copy sim data → MjData for one environment."""
    mj_data.qpos[:] = self._sim.data.qpos[env_idx].cpu().numpy()
    mj_data.qvel[:] = self._sim.data.qvel[env_idx].cpu().numpy()
    mj_data.ctrl[:] = self._sim.data.ctrl[env_idx].cpu().numpy()

  def _build_viewer(self) -> None:
    """Build MuJoCo passive viewer."""
    logging.debug("[Debug] _build_viewer called")
    try:
      import mujoco.viewer  # pylint: disable=g-import-not-at-top

      self._viewer = mujoco.viewer.launch_passive(
          self._mj_model, self._mj_data_display
      )
      logging.debug(f"[Debug] Viewer created: {self._viewer}")
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("WARNING: Could not create viewer: {}".format(e))
      self._viewer = None

  def _build_video_recorder(self) -> mjlab_recorder.MjlabVideoRecorder:
    logging.info("Video recording enabled")
    return mjlab_recorder.MjlabVideoRecorder(self)

  def _get_dof_pos_expmap(self, obj: ObjInfo) -> torch.Tensor:
    """Convert spherical joint quaternions to exp-map DOF representation."""
    num_envs = self._num_envs
    expmap = torch.zeros(
        num_envs, obj.num_dofs, device=self._device, dtype=torch.float32
    )

    # Copy non-sphere DOFs directly
    for q_idx, v_idx in zip(obj.nonsphere_q_indices, obj.nonsphere_v_indices):
      local_v = v_idx - obj.v_dof_start
      expmap[:, local_v] = self._sim.data.qpos[:, q_idx]

    # Convert sphere joint quaternions to exp-maps
    for q_start, v_start in zip(obj.sphere_q_starts, obj.sphere_v_starts):
      local_v = v_start - obj.v_dof_start
      q_wxyz = self._sim.data.qpos[:, q_start : q_start + 4]
      w = q_wxyz[:, 0:1]
      xyz = q_wxyz[:, 1:4]
      norm_xyz = torch.norm(xyz, dim=-1, keepdim=True).clamp(min=1e-8)
      angle = 2.0 * torch.atan2(norm_xyz, w)
      axis = xyz / norm_xyz
      exp_map = axis * angle
      expmap[:, local_v : local_v + 3] = exp_map

    return expmap

  def _set_dof_pos_with_sphere(
      self,
      env_id: int | torch.Tensor | None,
      obj: ObjInfo,
      dof_pos: torch.Tensor | int | float,
  ) -> None:
    """Set DOF positions when object has spherical joints."""
    qpos = self._sim.data.qpos

    if isinstance(dof_pos, (int, float)):
      if env_id is None:
        qpos[:, obj.q_dof_start : obj.q_dof_end] = dof_pos
      else:
        qpos[env_id, obj.q_dof_start : obj.q_dof_end] = dof_pos

    # Non-sphere DOFs
    for q_idx, v_idx in zip(obj.nonsphere_q_indices, obj.nonsphere_v_indices):
      local_v = v_idx - obj.v_dof_start
      if env_id is None:
        qpos[:, q_idx] = dof_pos[:, local_v]
      else:
        qpos[env_id, q_idx] = dof_pos[env_id, local_v]

    # Sphere DOFs: exp-map → quaternion
    for q_start, v_start in zip(obj.sphere_q_starts, obj.sphere_v_starts):
      local_v = v_start - obj.v_dof_start
      if env_id is None:
        exp_map = dof_pos[:, local_v : local_v + 3]
      else:
        exp_map = dof_pos[env_id, local_v : local_v + 3]
        if exp_map.dim() == 1:
          exp_map = exp_map.unsqueeze(0)

      angle = torch.norm(exp_map, dim=-1, keepdim=True)
      axis = exp_map / (angle + 1e-8)
      half_angle = angle * 0.5
      w = torch.cos(half_angle)
      xyz = axis * torch.sin(half_angle)

      mask = angle.squeeze(-1) < 1e-5
      w[mask] = 1.0
      xyz[mask] = 0.0

      q_wxyz = torch.cat([w, xyz], dim=-1)

      if env_id is None:
        qpos[:, q_start : q_start + 4] = q_wxyz
      else:
        qpos[env_id, q_start : q_start + 4] = q_wxyz

