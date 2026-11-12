"""Procedural Blender (Cycles) scenes for native high-frame-rate 3D-render clips. No outside assets.

Run inside Blender:
  blender -b --factory-startup -P scripts/hfr_blender_scene.py -- \
      --scene city --speed 1.0 --fps 60 --width 1920 --height 1080 --seconds 5 \
      --samples 48 --seed 7 --out /tmp/frames/ [--device GPU|CPU] [--night] [--motion-blur]
Writes PNG frames <out>/f_0001.png ... (8-bit RGB).
"""

from __future__ import annotations

import argparse
import colorsys
import math
import random
import sys

import bpy
from mathutils import Vector


def args_after_dashes():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--frames", type=int, default=0, help="override frame count (tests)")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--night", action="store_true")
    ap.add_argument("--motion-blur", action="store_true")
    ap.add_argument("--density", type=float, default=1.0)
    return ap.parse_args(argv)


A = args_after_dashes()
R = random.Random(A.seed)
N = A.frames or int(round(A.seconds * A.fps))
FPS = A.fps


# ----------------------------------------------------------------------------- setup
def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    if A.device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                devs = [d for d in prefs.devices if d.type == kind]
                if devs:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    break
            except TypeError:
                continue
        sc.cycles.device = "GPU"
    else:
        sc.cycles.device = "CPU"
    sc.cycles.samples = A.samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    # OpenImageDenoise on every device: the OptiX denoiser cannot be created on this host
    # (OptiX path tracing itself works); OIDN 2 runs on the GPU when one is in use.
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    if A.device == "GPU" and hasattr(sc.cycles, "denoising_use_gpu"):
        sc.cycles.denoising_use_gpu = True
    sc.cycles.max_bounces = 6
    # Keep scene/BVH/device data between frames: 58.1 s -> 26.7 s for 15 frames of the 5 s "objects" scene
    # (2026-11-10, CPU-throttled box); frames match within 1 level (PSNR 66.7 dB).
    sc.render.use_persistent_data = True
    sc.cycles.seed = A.seed
    sc.render.fps = FPS
    sc.render.fps_base = 1.0
    sc.frame_start, sc.frame_end = 1, N
    sc.render.resolution_x, sc.render.resolution_y = A.width, A.height
    sc.render.resolution_percentage = 100
    sc.render.use_motion_blur = A.motion_blur
    sc.render.motion_blur_shutter = 0.5
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.filepath = A.out.rstrip("/") + "/f_"
    sc.view_settings.view_transform = "AgX"
    for look in ("AgX - Medium High Contrast", "AgX - Punchy"):
        try:
            sc.view_settings.look = look
            break
        except TypeError:
            continue
    return sc


def world(sky=True, color=(0.05, 0.06, 0.08), strength=1.0, sun_elev=25.0):
    w = bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = strength
    if sky:
        try:
            tex = nt.nodes.new("ShaderNodeTexSky")
            tex.sky_type = "NISHITA"
            tex.sun_elevation = math.radians(sun_elev)
            tex.sun_rotation = math.radians(R.uniform(0, 360))
            nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
            bg.inputs["Strength"].default_value = 0.3 * strength
            return
        except Exception:  # noqa: BLE001
            pass
    bg.inputs["Color"].default_value = (*color, 1.0)


def mat(name, color=(0.8, 0.8, 0.8), metallic=0.0, rough=0.5, emission=None, estr=0.0, transmission=0.0, noise_bump=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]

    def setin(names, val):
        for nm in names:
            if nm in p.inputs:
                p.inputs[nm].default_value = val
                return

    setin(["Base Color"], (*color, 1.0))
    setin(["Metallic"], metallic)
    setin(["Roughness"], rough)
    setin(["Transmission Weight", "Transmission"], transmission)
    if emission is not None:
        setin(["Emission Color", "Emission"], (*emission, 1.0))
        setin(["Emission Strength"], estr)
    if noise_bump > 0:
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = R.uniform(20, 80)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = noise_bump
        nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m


def textured_mat(name, kind="noise", scale=8.0, c1=(0.2, 0.3, 0.1), c2=(0.5, 0.45, 0.3), rough=0.8, bump=0.3):
    """Detail-rich material: procedural texture -> colour ramp (+ bump)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new({"noise": "ShaderNodeTexNoise", "voronoi": "ShaderNodeTexVoronoi",
                        "wave": "ShaderNodeTexWave", "brick": "ShaderNodeTexBrick"}[kind])
    tex.inputs["Scale"].default_value = scale
    if kind == "noise":
        tex.inputs["Detail"].default_value = 12.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c1, 1)
    ramp.color_ramp.elements[1].color = (*c2, 1)
    out = tex.outputs["Fac"] if "Fac" in tex.outputs else tex.outputs[0]
    if kind == "voronoi":
        out = tex.outputs["Distance"]
    nt.links.new(out, ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = rough
    if bump > 0:
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        nt.links.new(out, b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], p.inputs["Normal"])
    return m


def add(prim, loc=(0, 0, 0), size=1.0, material=None, **kw):
    ops = {"cube": bpy.ops.mesh.primitive_cube_add, "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
           "ico": bpy.ops.mesh.primitive_ico_sphere_add, "torus": bpy.ops.mesh.primitive_torus_add,
           "cone": bpy.ops.mesh.primitive_cone_add, "cyl": bpy.ops.mesh.primitive_cylinder_add,
           "monkey": bpy.ops.mesh.primitive_monkey_add, "plane": bpy.ops.mesh.primitive_plane_add}
    if prim == "torus":
        ops[prim](location=loc, **kw)
    elif prim in ("cube", "monkey", "plane"):
        ops[prim](location=loc, size=size, **kw)
    elif prim == "cone":
        ops[prim](location=loc, radius1=size, radius2=0.0, **kw)
    else:
        ops[prim](location=loc, radius=size, **kw)
    ob = bpy.context.active_object
    if material is not None:
        ob.data.materials.append(material)
    return ob


def smooth(ob, level=1):
    for poly in ob.data.polygons:
        poly.use_smooth = True
    if level:
        md = ob.modifiers.new("sub", "SUBSURF")
        md.levels = md.render_levels = level


def linked_copy(ob, loc):
    c = ob.copy()
    c.location = loc
    bpy.context.collection.objects.link(c)
    return c


def sun(energy=3.0, elev=40, az=30, size=2.0):
    bpy.ops.object.light_add(type="SUN", rotation=(math.radians(90 - elev), 0, math.radians(az)))
    l = bpy.context.active_object
    l.data.energy = energy
    l.data.angle = math.radians(size)
    return l


def area(loc, energy=500, size=3.0, target=(0, 0, 0), color=(1, 1, 1)):
    bpy.ops.object.light_add(type="AREA", location=loc)
    l = bpy.context.active_object
    l.data.energy, l.data.size, l.data.color = energy, size, color
    look_at(l, target)
    return l


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def camera(lens=35.0):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens = lens
    cam.data.clip_end = 2000
    bpy.context.scene.camera = cam
    return cam


def track(cam, target_ob):
    c = cam.constraints.new("TRACK_TO")
    c.target = target_ob
    c.track_axis, c.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"


def empty(loc=(0, 0, 0)):
    bpy.ops.object.empty_add(location=loc)
    return bpy.context.active_object


def key_path(ob, points, frames=None, attr="location"):
    """Keyframe ``attr`` through ``points`` spread over the clip (Bezier, smooth)."""
    frames = frames or [1 + round(i * (N - 1) / (len(points) - 1)) for i in range(len(points))]
    for f, p in zip(frames, points):
        setattr(ob, attr, p)
        ob.keyframe_insert(data_path=attr, frame=f)


def linear_keys(ob):
    if ob.animation_data and ob.animation_data.action:
        act = ob.animation_data.action
        curves = getattr(act, "fcurves", None)
        if curves is None:  # layered actions (Blender 4.4+)
            curves = [fc for layer in act.layers for strip in layer.strips
                      for bag in strip.channelbags for fc in bag.fcurves]
        for fc in curves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def orbit(cam, center, radius, height, turns, phase=0.0):
    pts = []
    for i in range(9):
        a = phase + 2 * math.pi * turns * i / 8
        pts.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a), height))
    key_path(cam, pts)


# ----------------------------------------------------------------------------- scenes
def scene_product():
    world(sky=False, color=(0.02, 0.02, 0.025), strength=1.0)
    ob = add("monkey", (0, 0, 1.0), 1.2, mat("paint", (R.random(), R.random(), R.random()), 0.6, 0.25, noise_bump=0.05))
    smooth(ob, 2)
    add("cyl", (0, 0, 0), 1.6, mat("stand", (0.9, 0.9, 0.9), 0.0, 0.3), depth=0.3)
    add("plane", (0, 0, -0.15), 40, mat("floor", (0.15, 0.15, 0.17), 0, 0.4))
    area((3, -3, 4), 800, 3, (0, 0, 1))
    area((-4, -1, 3), 300, 4, (0, 0, 1), (0.8, 0.9, 1.0))
    area((0, 4, 3), 500, 2, (0, 0, 1), (1.0, 0.85, 0.7))
    key_path(ob, [(0, 0, 0), (0, 0, math.radians(360 * 0.6 * A.speed))], attr="rotation_euler")
    linear_keys(ob)
    cam = camera(50)
    tgt = empty((0, 0, 1))
    key_path(cam, [(0, -6.5, 2.2), (1.2, -6.0, 2.0)])
    track(cam, tgt)


def scene_rings():
    world(sky=False, color=(0.005, 0.005, 0.01), strength=1.0)
    cols = [(1, 0.2, 0.4), (0.2, 0.6, 1), (1, 0.7, 0.1), (0.4, 1, 0.5), (0.7, 0.3, 1)]
    n = int(14 * A.density)
    for i in range(n):
        r = 0.6 + i * 0.35
        ob = add("torus", (0, 0, 0), material=mat(f"r{i}", (0.05, 0.05, 0.05), 0.8, 0.2, cols[i % 5], R.uniform(2, 8)),
                 major_radius=r, minor_radius=0.04 + 0.02 * R.random())
        k = A.speed * R.uniform(0.5, 1.5) * (1 if i % 2 else -1)
        key_path(ob, [(R.uniform(0, 3), R.uniform(0, 3), 0), (R.uniform(0, 3) + 2.5 * k, R.uniform(0, 3) + 1.5 * k, 3 * k)], attr="rotation_euler")
        linear_keys(ob)
    cam = camera(40)
    tgt = empty((0, 0, 0))
    key_path(cam, [(0, -12, 3), (3, -10, 1)])
    track(cam, tgt)


def scene_objects():
    world(sky=True, strength=0.6, sun_elev=35)
    sun(3.0, 40, R.uniform(0, 360))
    add("plane", (0, 0, 0), 80, textured_mat("ground", "voronoi", 3.0, (0.08, 0.07, 0.06), (0.3, 0.28, 0.25), 0.6, 0.2))
    prims = ["sphere", "cube", "torus", "cone", "ico", "monkey", "cyl"]
    n = int(40 * A.density)
    for i in range(n):
        pr = R.choice(prims)
        x, y = R.uniform(-9, 9), R.uniform(-9, 9)
        s = R.uniform(0.3, 1.0)
        m = mat(f"m{i}", colorsys.hsv_to_rgb(R.random(), R.uniform(0.6, 1.0), R.uniform(0.25, 0.8)), R.choice([0, 0, 1]), R.uniform(0.05, 0.6),
                transmission=1.0 if R.random() < 0.12 else 0.0, noise_bump=0.1 if R.random() < 0.3 else 0.0)
        if pr == "torus":
            ob = add(pr, (x, y, s * 0.3), material=m, major_radius=s, minor_radius=s * 0.3)
        else:
            ob = add(pr, (x, y, s), s, m)
        smooth(ob, 1 if pr in ("sphere", "monkey", "torus") else 0)
        ob.rotation_euler = (R.uniform(0, 3), R.uniform(0, 3), R.uniform(0, 3))
    cam = camera(30)
    tgt = empty((0, 0, 0.5))
    orbit(cam, (0, 0), 16, 5.5, 0.35 * A.speed, R.uniform(0, 6))
    track(cam, tgt)


def scene_swarm():
    world(sky=False, color=(0.01, 0.012, 0.02), strength=1.0)
    area((0, -6, 8), 3000, 8, (0, 0, 0))
    area((6, 4, 2), 1200, 4, (0, 0, 0), (1.0, 0.6, 0.3))
    n = int(450 * A.density)
    bases = [add(p, (0, 0, -50), 0.12, mat(f"s{i}", (R.random(), R.random(), R.random()), R.choice([0, 1]), 0.3))
             for i, p in enumerate(["ico", "cube", "sphere"])]
    for i in range(n):
        ob = linked_copy(R.choice(bases), (0, 0, 0))
        r0, z0, ph = R.uniform(1, 6), R.uniform(-3, 3), R.uniform(0, 2 * math.pi)
        w = A.speed * R.uniform(0.6, 1.4) * (4.0 / (r0 + 1))
        pts = []
        for k in range(7):
            t = k / 6 * A.seconds
            a = ph + w * t
            pts.append((r0 * math.cos(a), r0 * math.sin(a), z0 + 0.8 * math.sin(a * 1.7 + ph)))
        key_path(ob, pts)
        key_path(ob, [(R.uniform(0, 6), R.uniform(0, 6), 0), (R.uniform(0, 6) + 8 * A.speed, R.uniform(0, 6) + 6 * A.speed, 4 * A.speed)], attr="rotation_euler")
    cam = camera(32)
    tgt = empty((0, 0, 0))
    key_path(cam, [(0, -14, 2), (-4, -12, 3.5), (2, -11, 1)])
    track(cam, tgt)


def scene_city():
    night = A.night
    world(sky=not night, color=(0.01, 0.012, 0.03), strength=0.2 if night else 0.8, sun_elev=20)
    if not night:
        sun(3.5, 30, R.uniform(0, 360))
    win = bpy.data.materials.new("facade")
    win.use_nodes = True
    nt = win.node_tree
    p = nt.nodes["Principled BSDF"]
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    add_xy = nt.nodes.new("ShaderNodeMath")
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    nt.links.new(sep.outputs["X"], add_xy.inputs[0])
    nt.links.new(sep.outputs["Y"], add_xy.inputs[1])
    nt.links.new(add_xy.outputs[0], comb.inputs["X"])
    nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
    brick = nt.nodes.new("ShaderNodeTexBrick")
    nt.links.new(comb.outputs[0], brick.inputs["Vector"])
    brick.inputs["Scale"].default_value = 0.9
    brick.inputs["Mortar Size"].default_value = 0.035
    brick.inputs["Brick Width"].default_value = 1.2
    brick.inputs["Row Height"].default_value = 0.5
    brick.offset = 0.0
    if night:
        brick.inputs["Color1"].default_value = (1.0, 0.78, 0.45, 1)
        brick.inputs["Color2"].default_value = (0.01, 0.01, 0.015, 1)
        brick.inputs["Mortar"].default_value = (0.02, 0.02, 0.025, 1)
        brick.inputs["Bias"].default_value = 0.55
        nt.links.new(brick.outputs["Color"], p.inputs["Emission Color"])
        p.inputs["Emission Strength"].default_value = 2.5
        p.inputs["Base Color"].default_value = (0.02, 0.02, 0.025, 1)
    else:
        brick.inputs["Color1"].default_value = (0.25, 0.32, 0.4, 1)
        brick.inputs["Color2"].default_value = (0.1, 0.13, 0.17, 1)
        brick.inputs["Mortar"].default_value = (0.55, 0.53, 0.5, 1)
        nt.links.new(brick.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Metallic"].default_value = 0.3
    p.inputs["Roughness"].default_value = 0.25
    road = textured_mat("road", "noise", 30.0, (0.04, 0.04, 0.045), (0.12, 0.12, 0.13), 0.7, 0.2)
    add("plane", (0, 60, 0), 400, road)
    base = add("cube", (0, 0, -100), 1.0, win)
    for gx in range(-6, 7):
        for gy in range(-2, int(30 * A.density)):
            if gx in (0,):
                continue
            h = R.uniform(4, 30) if abs(gx) > 1 else R.uniform(3, 12)
            ob = linked_copy(base, (gx * 8 + R.uniform(-1, 1), gy * 8, h))
            ob.scale = (R.uniform(2.5, 3.6), R.uniform(2.5, 3.6), h)
    if night:
        for i in range(60):
            add("sphere", (R.choice([-1.5, 1.5]), R.uniform(0, 240), 5.5), 0.15,
                mat(f"lamp{i}", (1, 1, 1), 0, 0.5, (1.0, 0.8, 0.5), 30.0))
    cam = camera(24)
    dist = 25 * A.speed * A.seconds
    key_path(cam, [(0, -10, 6), (0.8, -10 + dist * 0.5, 9), (-0.5, -10 + dist, 7)])
    tgt = empty((0, 0, 6))
    key_path(tgt, [(0, 40, 8), (1, 40 + dist * 0.5, 10), (0, 40 + dist, 8)])
    track(cam, tgt)
    linear_keys(cam)
    linear_keys(tgt)


def scene_terrain():
    world(sky=True, strength=1.0, sun_elev=12)
    sun(2.2, 14, R.uniform(0, 360))
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=400, y_subdivisions=400, size=300)
    ter = bpy.context.active_object
    tex = bpy.data.textures.new("hills", "CLOUDS")
    tex.noise_scale = 38.0
    tex.noise_depth = 5
    d = ter.modifiers.new("disp", "DISPLACE")
    d.texture = tex
    d.strength = 30.0
    for poly in ter.data.polygons:
        poly.use_smooth = True
    ter.data.materials.append(textured_mat("terrain", "noise", 40.0, (0.05, 0.09, 0.025), (0.22, 0.18, 0.12), 0.9, 0.5))
    bpy.ops.mesh.primitive_plane_add(size=300, location=(0, 0, -6))
    bpy.context.active_object.data.materials.append(mat("lake", (0.02, 0.06, 0.08), 0.0, 0.05))
    cam = camera(28)
    dist = 30 * A.speed * A.seconds
    key_path(cam, [(-dist / 2, -60, 32), (0, -60 + dist * 0.3, 26), (dist / 2, -60 + dist * 0.6, 30)])
    tgt = empty()
    key_path(tgt, [(-dist / 2 + 10, 0, 10), (5, dist * 0.3, 8), (dist / 2 + 10, dist * 0.6, 12)])
    track(cam, tgt)


def scene_ocean():
    world(sky=True, strength=1.0, sun_elev=8)
    sun(4.0, 8, R.uniform(0, 360), 1.0)
    bpy.ops.mesh.primitive_plane_add(size=2)
    sea = bpy.context.active_object
    oc = sea.modifiers.new("ocean", "OCEAN")
    oc.geometry_mode = "GENERATE"
    oc.resolution = 14
    oc.spatial_size = 80
    oc.repeat_x = oc.repeat_y = 4
    oc.wave_scale = 1.6 * A.speed
    oc.choppiness = 1.2
    oc.wind_velocity = 18
    oc.time = 1.0
    oc.keyframe_insert("time", frame=1)
    oc.time = 1.0 + A.seconds * A.speed
    oc.keyframe_insert("time", frame=N)
    linear_keys(sea)
    for poly in sea.data.polygons:
        poly.use_smooth = True
    sea.data.materials.append(mat("water", (0.004, 0.02, 0.03), 0.0, 0.06))
    cam = camera(30)
    key_path(cam, [(-20, -40, 6), (10 * A.speed, -30, 5), (20 * A.speed, -20, 6.5)])
    tgt = empty()
    key_path(tgt, [(0, 40, 0), (15, 45, 0), (30, 50, 2)])
    track(cam, tgt)


def scene_forest():
    world(sky=True, strength=0.9, sun_elev=30)
    sun(3.5, 35, R.uniform(0, 360), 3.0)
    add("plane", (0, 50, 0), 300, textured_mat("forest_floor", "noise", 60.0, (0.08, 0.1, 0.03), (0.3, 0.25, 0.12), 0.95, 0.6))
    trunk = add("cyl", (0, 0, -100), 0.25, mat("bark", (0.18, 0.12, 0.08), 0, 0.9, noise_bump=0.4), depth=4)
    crowns = [add("cone", (0, 0, -100), 1.6, textured_mat(f"leaf{i}", "voronoi", 25.0, (0.03, 0.12 + 0.05 * i, 0.03), (0.15, 0.35, 0.08), 0.8, 0.5), depth=5)
              for i in range(3)]
    for i in range(int(500 * A.density)):
        x, y = R.uniform(-30, 30), R.uniform(-5, 160)
        if abs(x) < 2.5:
            continue
        s = R.uniform(0.7, 1.4)
        t = linked_copy(trunk, (x, y, 2 * s))
        t.scale = (s, s, s)
        c = linked_copy(R.choice(crowns), (x, y, 4 * s + 2.5 * s))
        c.scale = (s, s, s * R.uniform(0.9, 1.4))
    cam = camera(26)
    dist = 12 * A.speed * A.seconds
    key_path(cam, [(0, -8, 2.2), (0.6, -8 + dist * 0.5, 2.6), (-0.4, -8 + dist, 2.0)])
    tgt = empty()
    key_path(tgt, [(0, 30, 3), (2, 30 + dist * 0.5, 3.5), (-1, 30 + dist, 3)])
    track(cam, tgt)
    linear_keys(cam)
    linear_keys(tgt)


# ----------------------------------------------------------------------------- scenes added 2026-11-10 (render set 2)
def bevel(ob, width=0.02):
    md = ob.modifiers.new("bevel", "BEVEL")
    md.width, md.segments = width, 2
    return ob


def scene_interior():
    """Archviz-style room: wood floor, furniture, bookshelf, sun through windows, slow camera dolly."""
    world(sky=True, strength=0.6, sun_elev=R.uniform(18, 32))
    sun(3.0, R.uniform(20, 32), R.uniform(160, 200), 1.0)
    white = mat("wall", (0.75, 0.73, 0.7), 0, 0.8)
    add("plane", (0, 0, 0), 12, textured_mat("wood", "wave", R.uniform(2, 4), (0.22, 0.11, 0.05), (0.45, 0.27, 0.13), 0.35, 0.05))
    add("plane", (0, 0, 3.2), 12, white)
    for loc, sc_ in [((0, 5, 1.6), (6, 0.1, 1.6)), ((-6, 0, 1.6), (0.1, 5, 1.6)), ((0, -5, 1.6), (6, 0.1, 1.6))]:
        ob = add("cube", loc, 1.0, white)
        ob.scale = sc_
    # window wall at +x: sill, lintel and pillars leave three openings
    for loc, sc_ in [((6, 0, 0.45), (0.1, 5, 0.45)), ((6, 0, 2.85), (0.1, 5, 0.35)), ((6, -4.4, 1.65), (0.1, 0.6, 0.75)),
                     ((6, -0.9, 1.65), (0.1, 0.35, 0.75)), ((6, 2.4, 1.65), (0.1, 0.35, 0.75)), ((6, 4.6, 1.65), (0.1, 0.4, 0.75))]:
        ob = add("cube", loc, 1.0, white)
        ob.scale = sc_
    area((0, 0, 3.1), 110, 4, (0, 0, 0), (1.0, 0.85, 0.7))
    rug = add("plane", (0.5, 0, 0.005), 1.0, textured_mat("rug", "brick", 6.0, (0.5, 0.1, 0.08), (0.85, 0.75, 0.55), 0.9, 0.1))
    rug.scale = (2.4, 1.6, 1)
    wood = mat("table", (0.3, 0.17, 0.08), 0, 0.3)
    top = bevel(add("cube", (0.5, 0, 0.75), 1.0, wood))
    top.scale = (1.1, 0.6, 0.04)
    for dx in (-1.0, 1.0):
        for dy in (-0.5, 0.5):
            add("cyl", (0.5 + dx, dy, 0.37), 0.04, wood, depth=0.74)
    fabric = textured_mat("fabric", "noise", 80.0, colorsys.hsv_to_rgb(R.random(), 0.5, 0.35), colorsys.hsv_to_rgb(R.random(), 0.4, 0.5), 0.9, 0.2)
    for loc, sc_ in [((-4.6, 0, 0.25), (0.6, 2.0, 0.25)), ((-5.1, 0, 0.75), (0.15, 2.0, 0.5)), ((-4.6, -1.9, 0.45), (0.6, 0.15, 0.2)), ((-4.6, 1.9, 0.45), (0.6, 0.15, 0.2))]:
        ob = bevel(add("cube", loc, 1.0, fabric), 0.06)
        ob.scale = sc_
    shelf = mat("shelf", (0.12, 0.08, 0.05), 0, 0.5)
    for k in range(5):
        ob = add("cube", (-2.0 + 0, 4.7, 0.3 + k * 0.55), 1.0, shelf)
        ob.scale = (1.6, 0.18, 0.02)
        x = -3.5
        while x < -0.5:
            w = R.uniform(0.04, 0.09)
            h = R.uniform(0.3, 0.45)
            b = add("cube", (x + w, 4.7, 0.32 + k * 0.55 + h / 2), 1.0, mat(f"book{k}_{x:.2f}", colorsys.hsv_to_rgb(R.random(), R.uniform(0.3, 0.8), R.uniform(0.2, 0.7)), 0, 0.6))
            b.scale = (w, 0.14, h / 2)
            x += 2 * w + 0.005
    for i in range(3):
        f = add("plane", (-5.9, -3 + i * 1.5, 1.9), 1.0, textured_mat(f"art{i}", R.choice(["noise", "voronoi", "wave"]), R.uniform(2, 6),
                                                                   colorsys.hsv_to_rgb(R.random(), 0.7, 0.6), colorsys.hsv_to_rgb(R.random(), 0.6, 0.9), 0.5, 0.0))
        f.rotation_euler = (math.radians(90), 0, math.radians(90))
        f.scale = (0.5, 0.35, 1)
    pot = add("cyl", (3.5, 3.8, 0.25), 0.25, mat("pot", (0.6, 0.3, 0.2), 0, 0.6), depth=0.5)
    for i in range(14):
        lf = add("sphere", (3.5 + R.uniform(-0.4, 0.4), 3.8 + R.uniform(-0.4, 0.4), 0.7 + R.uniform(0, 0.9)), R.uniform(0.12, 0.22),
                 textured_mat(f"leafi{i}", "voronoi", 30.0, (0.02, 0.15, 0.03), (0.1, 0.4, 0.08), 0.6, 0.3))
        lf.scale = (1, 0.5, 1.4)
    cam = camera(24)
    d = 2.2 * A.speed
    key_path(cam, [(-3.0, -4.3, 1.6), (-3.0 + d * 0.5, -4.1, 1.55), (-3.0 + d, -3.8, 1.5)])
    tgt = empty()
    key_path(tgt, [(0.5, 2.0, 0.9), (1.5, 1.0, 1.0), (2.0 + d * 0.3, 0.0, 1.1)])
    track(cam, tgt)


def scene_space():
    """Star field, banded planet and an asteroid field the camera flies through."""
    w = bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 260.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
    ramp.color_ramp.elements[1].position = 0.07
    ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
    nt.links.new(tc.outputs["Generated"], vor.inputs["Vector"])
    nt.links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 3.0
    sun(5.0, R.uniform(10, 30), R.uniform(-55, 55), 0.5)  # az near 0 = light from behind the camera
    sun(0.6, 20, 180, 5.0).data.color = (0.5, 0.65, 1.0)  # dim blue rim/fill
    planet = add("sphere", (R.uniform(-30, 30), 160, R.uniform(-25, 10)), 45, textured_mat(
        "planet", "wave", 3.0, colorsys.hsv_to_rgb(R.random(), 0.5, 0.35), colorsys.hsv_to_rgb(R.random(), 0.35, 0.8), 0.7, 0.05), segments=96, ring_count=48)
    smooth(planet, 0)
    key_path(planet, [(0.3, 0, 0), (0.3, 0, 0.15 * A.speed)], attr="rotation_euler")
    rock = textured_mat("rock", "noise", 6.0, (0.08, 0.07, 0.06), (0.35, 0.31, 0.27), 0.9, 0.8)
    bases = []
    for i in range(3):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0, location=(0, 0, -500))
        b = bpy.context.active_object
        t = bpy.data.textures.new(f"ast{i}", "CLOUDS")
        t.noise_scale = R.uniform(0.5, 1.0)
        dmod = b.modifiers.new("disp", "DISPLACE")
        dmod.texture, dmod.strength = t, R.uniform(0.35, 0.6)
        b.scale = (1, R.uniform(0.6, 0.9), R.uniform(0.5, 0.8))
        b.data.materials.append(rock)
        smooth(b, 0)
        bases.append(b)
    dist = 18 * A.speed * A.seconds
    for i in range(int(160 * A.density)):
        ob = linked_copy(R.choice(bases), (R.uniform(-25, 25), R.uniform(0, dist + 60), R.uniform(-12, 12)))
        if abs(ob.location.x) < 2.5 and abs(ob.location.z) < 2.5:
            ob.location.x += 6
        s = R.uniform(0.3, 2.5) * (1.6 if R.random() < 0.1 else 1.0)
        ob.scale = (s, s * R.uniform(0.6, 1.0), s * R.uniform(0.5, 0.9))
        key_path(ob, [(R.uniform(0, 6), R.uniform(0, 6), 0), (R.uniform(0, 6) + A.speed * R.uniform(-1.5, 1.5), R.uniform(0, 6) + A.speed * R.uniform(-1, 1), 0)], attr="rotation_euler")
        linear_keys(ob)
    cam = camera(28)
    key_path(cam, [(0, -10, 0), (1.5, -10 + dist * 0.5, 1.0), (-1, -10 + dist, 0.5)])
    tgt = empty()
    key_path(tgt, [(0, 40, 0), (2, 40 + dist * 0.5, 1.5), (-1, 40 + dist, 0)])
    track(cam, tgt)
    linear_keys(cam)
    linear_keys(tgt)


def toon_mat(name, color):
    """Flat cartoon look in Cycles: diffuse base + self-emission of the same colour (shadows stay readable)."""
    return mat(name, color, 0.0, 1.0, color, 0.45)


def scene_toon():
    """Cartoon / game-like: flat saturated colours, black Freestyle outlines, side-scrolling camera."""
    sc = bpy.context.scene
    sc.view_settings.view_transform = "Standard"
    try:
        sc.view_settings.look = "None"
    except TypeError:
        pass
    sc.render.use_freestyle = True
    sc.render.line_thickness_mode = "ABSOLUTE"
    sc.render.line_thickness = 1.0
    vl = bpy.context.view_layer
    fs = vl.freestyle_settings
    ls = fs.linesets[0] if len(fs.linesets) else fs.linesets.new("toon")
    if ls.linestyle is None:
        ls.linestyle = bpy.data.linestyles.new("toon_line")
    ls.linestyle.thickness = 2.2 * A.height / 1080
    ls.linestyle.color = (0.02, 0.02, 0.03)
    world(sky=False, color=(0.45, 0.7, 1.0), strength=1.0)
    sun(2.5, 50, R.uniform(120, 240), 0.5)
    add("plane", (0, 0, 0), 600, toon_mat("grass", (0.25, 0.65, 0.2)))
    span = 60 * max(1.0, A.speed) + 40
    path = add("plane", (span / 2, 3, 0.01), 1.0, toon_mat("path", (0.75, 0.6, 0.38)))
    path.scale = (span, 1.6, 1)
    for i in range(int(60 * A.density)):
        x, y = R.uniform(-15, span), R.choice([R.uniform(-9, 0.8), R.uniform(5.2, 8)])
        if R.random() < 0.4:
            add("sphere", (x, y, 0.3), R.uniform(0.4, 0.8), toon_mat(f"bush{i}", colorsys.hsv_to_rgb(R.uniform(0.25, 0.4), 0.7, 0.45))).scale = (1.3, 1, 0.8)
        else:
            add("cyl", (x, y, 0.25), 0.03, toon_mat(f"stem{i}", (0.2, 0.5, 0.15)), depth=0.5)
            add("sphere", (x, y, 0.55), 0.12, toon_mat(f"flower{i}", colorsys.hsv_to_rgb(R.random(), 0.75, 1.0)))
    for i in range(int(18 * A.density)):
        h = add("sphere", (R.uniform(-20, span), R.uniform(25, 60), -6), R.uniform(10, 18), toon_mat(f"hill{i}", colorsys.hsv_to_rgb(R.uniform(0.25, 0.38), 0.6, R.uniform(0.45, 0.7))))
        h.scale = (1.6, 1, 0.8)
    for i in range(int(14 * A.density)):
        x, y = R.uniform(-15, span), R.uniform(6, 18)
        add("cyl", (x, y, 1.2), 0.25, toon_mat(f"trunk{i}", (0.45, 0.28, 0.12)), depth=2.4)
        add("sphere", (x, y, 3.2), R.uniform(1.0, 1.6), toon_mat(f"crown{i}", colorsys.hsv_to_rgb(R.uniform(0.22, 0.4), 0.75, 0.6)))
    for i in range(int(7 * A.density)):
        x, y = R.uniform(-10, span), R.uniform(9, 20)
        hc = colorsys.hsv_to_rgb(R.random(), 0.55, 0.9)
        hb = add("cube", (x, y, 1.5), 1.5, toon_mat(f"house{i}", hc))
        hb.scale = (1.2, 1, 1)
        roof = add("cone", (x, y, 3.8), 2.5, toon_mat(f"roof{i}", (0.8, 0.2, 0.15)), depth=1.6, vertices=4)
        roof.rotation_euler = (0, 0, math.radians(45))
    for i in range(int(9 * A.density)):
        add("sphere", (R.uniform(-10, span), R.uniform(30, 50), R.uniform(12, 20)), R.uniform(1.5, 3), toon_mat(f"cloud{i}", (1, 1, 1))).scale = (2.2, 1, 0.8)
    # characters: capsule bodies hopping along the path
    for i in range(int(6 * A.density)):
        col = colorsys.hsv_to_rgb(R.random(), 0.8, 0.95)
        body = add("sphere", (0, 0, 0), 0.6, toon_mat(f"char{i}", col))
        body.scale = (0.8, 0.8, 1.1)
        eye = add("sphere", (0, 0, 0), 0.12, toon_mat(f"eye{i}", (1, 1, 1)))
        eye.parent = body
        eye.location = (0.25, -0.62, 0.35)
        x0, v = R.uniform(-5, span * 0.6), A.speed * R.uniform(2, 6) * R.choice([1, 1, -1])
        y = R.uniform(1, 5)
        hop = R.uniform(1.5, 3.0)
        pts = []
        for k in range(int(A.seconds * hop * 4) + 1):
            t = k / (hop * 4)
            pts.append((x0 + v * t, y, 0.7 + 1.2 * abs(math.sin(math.pi * hop * t))))
        key_path(body, pts, frames=[1 + round(k / (hop * 4) * FPS) for k in range(len(pts))])
    cam = camera(35)
    d = 8 * A.speed * A.seconds
    key_path(cam, [(0, -22, 4), (d, -22, 4.3)])
    linear_keys(cam)
    cam.rotation_euler = (math.radians(84), 0, 0)


def scene_snow():
    """Dusk pine forest in heavy snowfall with warm lamps; small fast falling flakes (+ motion blur)."""
    world(sky=True, strength=0.6, sun_elev=6)
    sun(0.9, 7, R.uniform(0, 360), 3.0)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=200, y_subdivisions=200, size=200, location=(0, 60, 0))
    gr = bpy.context.active_object
    t = bpy.data.textures.new("drifts", "CLOUDS")
    t.noise_scale = 6.0
    dm = gr.modifiers.new("disp", "DISPLACE")
    dm.texture, dm.strength = t, 1.2
    smooth(gr, 0)
    gr.data.materials.append(mat("snowground", (0.9, 0.94, 1.0), 0, 0.5, noise_bump=0.3))
    trunk = add("cyl", (0, 0, -100), 0.2, mat("bark2", (0.12, 0.08, 0.05), 0, 0.9), depth=3)
    crowns = [add("cone", (0, 0, -100), 1.5, textured_mat(f"pine{i}", "noise", 25.0, (0.01, 0.04 + 0.015 * i, 0.02), (0.05, 0.13, 0.06), 0.8, 0.4), depth=5)
              for i in range(3)]
    for i in range(int(260 * A.density)):
        x, y = R.uniform(-35, 35), R.uniform(0, 120)
        if abs(x) < 3:
            continue
        s = R.uniform(0.7, 1.5)
        linked_copy(trunk, (x, y, 1.5 * s)).scale = (s, s, s)
        c = linked_copy(R.choice(crowns), (x, y, 3 * s + 2.5 * s))
        c.scale = (s, s, s)
    for i in range(10):
        y = 6 + i * 10
        add("cyl", (2.8, y, 1.6), 0.06, mat(f"post{i}", (0.05, 0.05, 0.05), 1, 0.4), depth=3.2)
        add("sphere", (2.8, y, 3.3), 0.18, mat(f"lampw{i}", (1, 1, 1), 0, 0.5, (1.0, 0.7, 0.35), 25.0))
        bpy.ops.object.light_add(type="POINT", location=(2.8, y, 3.0))
        bpy.context.active_object.data.energy = 120
        bpy.context.active_object.data.color = (1.0, 0.7, 0.4)
    flake = add("ico", (0, 0, -100), 1.0, mat("flake", (1, 1, 1), 0, 0.4, (0.9, 0.92, 1.0), 0.6), subdivisions=1)
    dist = 2.0 * A.speed * A.seconds
    for i in range(int(2200 * A.density)):
        x, y = R.uniform(-7, 7), R.uniform(-6 + dist * 0.3, 26 + dist)
        fall = A.speed * R.uniform(1.0, 2.2) * A.seconds
        z0 = R.uniform(0, 8 + fall)
        drift = A.speed * R.uniform(0.3, 1.2)
        f = linked_copy(flake, (x, y, z0))
        sz = R.uniform(0.012, 0.035)
        f.scale = (sz, sz, sz)
        key_path(f, [(x, y, z0), (x + drift, y + R.uniform(-0.3, 0.3), z0 - fall)])
        linear_keys(f)
    cam = camera(32)
    key_path(cam, [(0, -8, 1.8), (0.3, -8 + dist * 0.5, 1.9), (0, -8 + dist, 1.8)])
    tgt = empty()
    key_path(tgt, [(0, 30, 2.5), (0.5, 30 + dist * 0.5, 2.7), (0, 30 + dist, 2.5)])
    track(cam, tgt)


def scene_robots():
    """Character animation: walking robots (articulated primitives, swinging limbs) crossing a tiled plaza."""
    world(sky=True, strength=0.45, sun_elev=R.uniform(30, 50))
    sun(2.2, R.uniform(35, 50), R.uniform(0, 360), 1.5)
    add("plane", (0, 0, 0), 120, textured_mat("tiles", "brick", 12.0, (0.14, 0.13, 0.12), (0.3, 0.27, 0.24), 0.6, 0.15))
    for i in range(10):
        a = 2 * math.pi * i / 10
        b = add("cube", (22 * math.cos(a), 22 * math.sin(a), 4), 1.0, mat(f"bld{i}", colorsys.hsv_to_rgb(R.random(), 0.25, 0.6), 0, 0.6))
        b.scale = (3, 3, R.uniform(3, 8))
    for i in range(6):
        add("cyl", (R.uniform(-10, 10), R.uniform(-10, 10), 2), 0.35, mat(f"pil{i}", (0.8, 0.8, 0.78), 0, 0.4), depth=4)

    def limb(parent, pos, length, radius, m):
        piv = empty((0, 0, 0))
        piv.parent = parent
        piv.location = pos
        seg = add("cyl", (0, 0, 0), radius, m, depth=length)
        seg.parent = piv
        seg.location = (0, 0, -length / 2)
        return piv

    for i in range(int(8 * A.density)):
        col = colorsys.hsv_to_rgb(R.random(), 0.7, 0.8)
        m = mat(f"robot{i}", col, R.choice([0.0, 0.8]), 0.3)
        dark = mat(f"joint{i}", (0.05, 0.05, 0.06), 0.8, 0.3)
        root = empty((0, 0, 0))
        body = bevel(add("cube", (0, 0, 0), 1.0, m), 0.04)
        body.parent = root
        body.location = (0, 0, 1.35)
        body.scale = (0.32, 0.22, 0.42)
        head = add("sphere", (0, 0, 0), 0.2, m)
        head.parent = root
        head.location = (0, 0, 2.0)
        visor = add("cube", (0, 0, 0), 1.0, mat(f"visor{i}", (0.0, 0.0, 0.0), 0, 0.2, (0.2, 0.9, 1.0), 4.0))
        visor.parent = root
        visor.location = (0, -0.17, 2.02)
        visor.scale = (0.13, 0.04, 0.04)
        legs = [limb(root, (sx * 0.14, 0, 0.95), 0.95, 0.07, dark) for sx in (-1, 1)]
        arms = [limb(root, (sx * 0.4, 0, 1.7), 0.75, 0.05, dark) for sx in (-1, 1)]
        heading = R.uniform(0, 2 * math.pi)
        v = A.speed * R.uniform(0.8, 1.4)
        start = Vector((R.uniform(-5, 5), R.uniform(-5, 5), 0)) - Vector((math.cos(heading), math.sin(heading), 0)) * v * A.seconds / 2
        root.rotation_euler = (0, 0, heading + math.pi / 2)
        key_path(root, [tuple(start), tuple(start + Vector((math.cos(heading), math.sin(heading), 0)) * v * A.seconds)])
        linear_keys(root)
        fstep = v / 0.9  # strides per second
        nk = int(A.seconds * fstep * 4) + 1
        frames = [1 + round(k / (fstep * 4) * FPS) for k in range(nk)]
        amp = math.radians(28)
        for j, (lg, ar) in enumerate(zip(legs, arms)):
            ph = math.pi * j
            key_path(lg, [(amp * math.sin(ph + math.pi / 2 * k), 0, 0) for k in range(nk)], frames=frames, attr="rotation_euler")
            key_path(ar, [(-amp * math.sin(ph + math.pi / 2 * k), 0, 0) for k in range(nk)], frames=frames, attr="rotation_euler")
    cam = camera(35)
    tgt = empty((0, 0, 1.2))
    orbit(cam, (0, 0), 9, 2.6, 0.12 * A.speed, R.uniform(0, 6))
    track(cam, tgt)


SCENES = {"product": scene_product, "rings": scene_rings, "objects": scene_objects, "swarm": scene_swarm,
          "city": scene_city, "terrain": scene_terrain, "ocean": scene_ocean, "forest": scene_forest,
          "interior": scene_interior, "space": scene_space, "toon": scene_toon, "snow": scene_snow, "robots": scene_robots}

setup_scene()
SCENES[A.scene]()
bpy.ops.render.render(animation=True)
print("RENDER DONE", N, "frames", flush=True)
