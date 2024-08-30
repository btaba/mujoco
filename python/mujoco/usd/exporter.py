# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""USD exporter."""

import os
from typing import List, Optional

import mujoco
import mujoco.usd.camera as camera_module
import mujoco.usd.lights as light_module
import mujoco.usd.objects as object_module
import mujoco.usd.shapes as shapes_module
import numpy as np
from PIL import Image as im
from PIL import ImageOps

# TODO: b/288149332 - Remove once USD Python Binding works well with pytype.
# pytype: disable=module-attr
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom


class USDExporter:
  """MuJoCo to USD exporter for porting scenes to external renderers."""

  def __init__(
      self,
      model: mujoco.MjModel,
      height: int = 480,
      width: int = 480,
      max_geom: int = 10000,
      output_directory_name: str = "mujoco_usdpkg",
      output_directory_root: str = "./",
      light_intensity: int = 10000,
      camera_names: Optional[List[str]] = None,
      specialized_materials_file: Optional[str] = None,
      verbose: bool = True,
  ):
    """Initializes a new USD Exporter.

    Args:
        model: an MjModel instance.
        height: image height in pixels.
        width: image width in pixels.
        max_geom: optional integer specifying the maximum number of geoms that
          can be rendered in the same scene. If None this will be chosen
          automatically based on the estimated maximum number of renderable
          geoms in the model.
        output_directory_name: name of root directory to store outputted frames
          and assets generated by the USD renderer.
        output_directory_root: path to root directory storing generated frames
          and assets by the USD renderer.
        light_intensity: intensity of the light in the scene.
        camera_names: list of camera names to be used in the scene.
        specialized_materials_file: path to a file containing a list of
          materials to be used in the scene.
        verbose: decides whether to print updates.
    """

    buffer_width = model.vis.global_.offwidth
    buffer_height = model.vis.global_.offheight

    if width > buffer_width:
      raise ValueError(f"""
                Image width {width} > framebuffer width {buffer_width}. Either reduce the image
                width or specify a larger offscreen framebuffer in the model XML using the
                clause:
                <visual>
                <global offwidth="my_width"/>
                </visual>""".lstrip())

    if height > buffer_height:
      raise ValueError(f"""
                Image height {height} > framebuffer height {buffer_height}. Either reduce the
                image height or specify a larger offscreen framebuffer in the model XML using
                the clause:
                <visual>
                <global offheight="my_height"/>
                </visual>""".lstrip())

    self.model = model
    self.height = height
    self.width = width
    self.max_geom = max_geom
    self.output_directory_name = output_directory_name
    self.output_directory_root = output_directory_root
    self.light_intensity = light_intensity
    self.camera_names = camera_names
    self.specialized_materials_file = specialized_materials_file
    self.verbose = verbose

    self.frame_count = 0  # maintains how many times we have saved the scene
    self.updates = 0

    self.geom_names = set()
    self.geom_refs = {}

    # initializing list of lights and cameras
    self.usd_lights = []
    self.usd_cameras = []

    # initializing rendering requirements
    self.renderer = mujoco.Renderer(model, height, width, max_geom)
    self._initialize_usd_stage()
    self._scene_option = mujoco.MjvOption()  # using default scene option

    # initializing output_directories
    self._initialize_output_directories()

    # loading required textures for the scene
    self._load_textures()

  @property
  def usd(self):
    """Returns the USD file as a string."""
    return self.stage.GetRootLayer().ExportToString()

  @property
  def scene(self):
    """Returns the scene."""
    return self.renderer.scene

  def _initialize_usd_stage(self):
    """Initializes a USD stage to represent the mujoco scene."""
    self.stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
    self.stage.SetStartTimeCode(0)
    # add as user input
    self.stage.SetTimeCodesPerSecond(60.0)

    default_prim = UsdGeom.Xform.Define(
        self.stage, Sdf.Path("/World")
    ).GetPrim()
    self.stage.SetDefaultPrim(default_prim)

  def _initialize_output_directories(self):
    """Initializes output directories to store frames and assets."""
    self.output_directory_path = os.path.join(
        self.output_directory_root, self.output_directory_name
    )
    if not os.path.exists(self.output_directory_path):
      os.makedirs(self.output_directory_path)

    self.frames_directory = os.path.join(self.output_directory_path, "frames")
    if not os.path.exists(self.frames_directory):
      os.makedirs(self.frames_directory)

    self.assets_directory = os.path.join(self.output_directory_path, "assets")
    if not os.path.exists(self.assets_directory):
      os.makedirs(self.assets_directory)

    if self.verbose:
      print("Writing output frames and assets to"
            f" {self.output_directory_path}",
            "green"
      )

  def update_scene(
      self,
      data: mujoco.MjData,
      scene_option: Optional[mujoco.MjvOption] = None,
  ):
    """Updates the scene with latest sim data.

    Args:
        data: structure storing current simulation state
        scene_option: we use this to determine which geom groups to activate
    """

    self.frame_count += 1

    scene_option = scene_option or self._scene_option

    # update the mujoco renderer
    self.renderer.update_scene(data, scene_option=scene_option)

    if self.updates == 0:
      self._initialize_usd_stage()
      self._load_lights()
      self._load_cameras()

    self._update_geoms()
    self._update_lights()
    self._update_cameras(data, scene_option=scene_option)

    self.updates += 1

  def _load_textures(self):
    """Load textures."""
    data_adr = 0
    self.texture_files = []
    for texture_id in range(self.model.ntex):
      texture_height = self.model.tex_height[texture_id]
      texture_width = self.model.tex_width[texture_id]
      texture_nchannel = self.model.tex_nchannel[texture_id]
      pixels = texture_nchannel * texture_height * texture_width
      img = im.fromarray(
          self.model.tex_data[data_adr : data_adr + pixels].reshape(
              texture_height, texture_width, 3
          )
      )
      img = ImageOps.flip(img)

      texture_file_name = f"texture_{texture_id}.png"

      img.save(os.path.join(self.assets_directory, texture_file_name))

      relative_path = os.path.relpath(
          self.assets_directory, self.frames_directory
      )
      img_path = os.path.join(
          relative_path, texture_file_name
      )

      self.texture_files.append(img_path)

      data_adr += pixels

    if self.verbose:
      print(f"Completed writing {self.model.ntex} textures to"
            f" {self.assets_directory}",
            "green",
      )

  def _load_geom(self, geom: mujoco.MjvGeom):
    """Loads a geom into the USD scene."""
    geom_name = self._get_geom_name(geom)

    assert geom_name not in self.geom_names

    if geom.matid == -1:
      geom_textures = []
    else:
      geom_textures = [
          (self.texture_files[i], self.model.tex_type[i]) if i != -1 else None
          for i in self.model.mat_texid[geom.matid]
      ]

    # handling meshes in our scene
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
      usd_geom = object_module.USDMesh(
          stage=self.stage,
          model=self.model,
          geom=geom,
          obj_name=geom_name,
          dataid=self.model.geom_dataid[geom.objid],
          rgba=geom.rgba,
          geom_textures=geom_textures,
      )
    else:
      # handling tendons in our scene
      if geom.objtype == mujoco.mjtObj.mjOBJ_TENDON:
        mesh_config = shapes_module.mesh_config_generator(
            name=geom_name,
            geom_type=geom.type,
            size=np.array([1.0, 1.0, 1.0]),
            decouple=True
        )
        usd_geom = object_module.USDTendon(
            mesh_config=mesh_config,
            stage=self.stage,
            model=self.model,
            geom=geom,
            obj_name=geom_name,
            rgba=geom.rgba,
            geom_textures=geom_textures,
        )
      # handling primitives in our scene
      else:
        mesh_config = shapes_module.mesh_config_generator(
            name=geom_name,
            geom_type=geom.type,
            size=geom.size
        )
        usd_geom = object_module.USDPrimitiveMesh(
            mesh_config=mesh_config,
            stage=self.stage,
            model=self.model,
            geom=geom,
            obj_name=geom_name,
            rgba=geom.rgba,
            geom_textures=geom_textures,
        )

    self.geom_names.add(geom_name)
    self.geom_refs[geom_name] = usd_geom

  def _update_geoms(self):
    """Iterate through all geoms in the scene and makes update."""
    for i in range(self.scene.ngeom):
      geom = self.scene.geoms[i]
      geom_name = self._get_geom_name(geom)

      if geom_name not in self.geom_names:
        # load a new object into USD
        self._load_geom(geom)

      if geom.objtype == mujoco.mjtObj.mjOBJ_TENDON:
        tendon_scale = geom.size
        self.geom_refs[geom_name].update(
            pos=geom.pos,
            mat=geom.mat,
            scale=tendon_scale,
            visible=geom.rgba[3] > 0,
            frame=self.updates,
        )
      else:
        self.geom_refs[geom_name].update(
            pos=geom.pos,
            mat=geom.mat,
            visible=geom.rgba[3] > 0,
            frame=self.updates,
        )

  def _load_lights(self):
    # initializes an usd light object for every light in the scene
    for i in range(self.scene.nlight):
      light = self.scene.lights[i]
      if not np.allclose(light.pos, [0, 0, 0]):
        self.usd_lights.append(
            light_module.USDSphereLight(stage=self.stage, obj_name=str(i))
        )
      else:
        self.usd_lights.append(None)

  def _update_lights(self):
    for i in range(self.scene.nlight):
      light = self.scene.lights[i]

      if np.allclose(light.pos, [0, 0, 0]):
        continue

      if i >= len(self.usd_lights) or self.usd_lights[i] is None:
        continue

      self.usd_lights[i].update(
          pos=light.pos,
          intensity=self.light_intensity,
          color=light.diffuse,
          frame=self.updates,
      )

  def _load_cameras(self):
    if self.camera_names is not None:
      for name in self.camera_names:
        self.usd_cameras.append(
            camera_module.USDCamera(stage=self.stage, obj_name=name))

  def _update_cameras(
      self,
      data: mujoco.MjData,
      scene_option: Optional[mujoco.MjvOption] = None,
  ):
    """Updates cameras.

    Args:
      data: An MjData instance.
      scene_option: An optional MjvOption instance.
    """
    for i in range(len(self.usd_cameras)):
      camera = self.usd_cameras[i]
      camera_name = self.camera_names[i]

      self.renderer.update_scene(
          data, scene_option=scene_option, camera=camera_name
      )

      avg_camera = mujoco.mjv_averageCamera(
          self.scene.camera[0], self.scene.camera[1])

      forward = avg_camera.forward
      up = avg_camera.up
      right = np.cross(forward, up)

      rotation = np.eye(3)
      rotation[:, 0] = right
      rotation[:, 1] = up
      rotation[:, 2] = -forward

      camera.update(
          cam_pos=avg_camera.pos, cam_mat=rotation, frame=self.updates
      )

  def add_light(
      self,
      pos: List[float],
      intensity: int,
      radius: Optional[float] = 1.0,
      color: Optional[np.ndarray] = np.array([0.3, 0.3, 0.3]),
      obj_name: Optional[str] = "light_1",
      light_type: Optional[str] = "sphere",
  ):
    """Adds a user defined, fixed light.

    Args:
        pos: position of the light in 3D space.
        intensity: intensity of the light.
        radius: radius of the light to be used by renderer.
        color: color of the light.
        obj_name: name associated with the light.
        light_type: type of light (sphere or dome).
    """
    if light_type == "sphere":
      new_light = light_module.USDSphereLight(
          stage=self.stage, obj_name=obj_name, radius=radius
      )

      new_light.update(
          pos=np.array(pos), intensity=intensity, color=color, frame=0
      )
    elif light_type == "dome":
      new_light = light_module.USDDomeLight(stage=self.stage, obj_name=obj_name)
      new_light.update(intensity=intensity, color=color)

  def add_camera(
      self,
      pos: List[float],
      rotation_xyz: List[float],
      obj_name: Optional[str] = "camera_1",
  ):
    """Adds a user defined, fixed camera.

    Args:
        pos: position of the camera in 3D space.
        rotation_xyz: euler rotation of the camera.
        obj_name: name associated with the camera.
    """
    new_camera = camera_module.USDCamera(
        stage=self.stage, obj_name=obj_name)

    R = np.zeros(9)
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, rotation_xyz, "xyz")
    mujoco.mju_quat2Mat(R, quat)
    new_camera.update(cam_pos=np.array(pos), cam_mat=R, frame=0)

  def save_scene(self, filetype: str = "usd"):
    """Saves the scene to a USD file."""
    assert filetype in ["usd", "usda", "usdc"]
    self.stage.SetEndTimeCode(self.frame_count)

    # post-processing for visibility of geoms in scene
    for _, geom_ref in self.geom_refs.items():
      geom_ref.update_visibility(False, geom_ref.last_visible_frame+1)

    self.stage.Export(
        f"{self.output_directory_root}/{self.output_directory_name}/" +
        f"frames/frame_{self.frame_count}.{filetype}"
    )
    if self.verbose:
      print(
        f"Completed writing frame_{self.frame_count}.{filetype}", "green"
      )

  def _get_geom_name(self, geom):
    """Adding id as part of name for USD file."""
    geom_name = mujoco.mj_id2name(self.model, geom.objtype, geom.objid)
    if not geom_name:
      geom_name = "None"
    geom_name += f"_id{geom.objid}"

    # adding additional naming information to differentiate
    # between geoms and tendons
    if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
      geom_name += "_geom"
    elif geom.objtype == mujoco.mjtObj.mjOBJ_TENDON:
      geom_name += f"_tendon_segid{geom.segid}"

    return geom_name

  # for debugging purposes, prints all geoms in scene
  # including those part of tendons
  def _print_scene_geom_info(self):
    for i in range(self.scene.ngeom):
      geom = self.scene.geoms[i]
      geom_name = self._get_geom_name(geom)
      print(i, geom_name)
