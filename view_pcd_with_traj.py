#!/usr/bin/env python3
# - Trajectory format: timestamp x y z qx qy qz qw  (C2W)
# - Uses synthetic intrinsics from HFOV + image size for frustum shape
# - Global transform (Rg,tg) + BEV transform (Rb,tb)
# - Crop ceilings for BEV view: remove top X cm along +Z in BEV frame

import re, sys
import signal
from pathlib import Path
import numpy as np
import open3d as o3d


# ========= INPUT PATHS =========
PLY_PATH  = "logs/Navigation_reloc.ply"
TRAJ_PATH = "logs/Navigation_reloc.txt"

# ========= VIS PARAMS =========
IMG_W, IMG_H = 640, 480     # only used to shape the frustum
HFOV_DEG     = 90.0         # horizontal field-of-view (deg) for frustum shape
FRUSTUM_DEPTH = 0.5         # frustum length in scene units (e.g., meters)
DRAW_EVERY    = 1           # draw every Nth pose
FRUSTUM_COLOR = (0.9, 0.3, 0.1)
TRAJ_COLOR    = (0.0, 1.0, 0.0)
WORLD_AXIS_SIZE = 0.3

# ========= GLOBAL TRANSFORM =========
# Apply a global SE(3):  X' = Rg * X + tg
# Default: 180° about X -> Rg = diag(1,-1,-1) (turn scene upside-down or upright)
ROT_AXIS = "x"              # 'x' | 'y' | 'z' | 'custom'
ROT_DEG  = 180.0            # rotation degrees (ignored if custom)
TG       = np.array([0.0, 0.0, 0.0])  # optional global translation
RG_CUSTOM = np.eye(3)       # used if ROT_AXIS == "custom"

# ========= BEV TRANSFORM =========
APPLY_BEV    = True         # enable/disable BEV transform
BEV_ROT_AXIS = "x"          # 'x' | 'y' | 'z' | 'custom'
BEV_ROT_DEG  = 90.0        # look-from-above default
TB           = np.array([0.0, 0.0, 0.0])  # optional BEV translation
RB_CUSTOM    = np.eye(3)    # used if BEV_ROT_AXIS == "custom"

# ========= CROP (in BEV frame) =========
# Remove the top X cm after BEV transform.
CROP_TOP_CM = 135.0

def rodrigues(axis, deg):
    th = np.deg2rad(deg)
    if axis == "x":
        return np.array([[1,0,0],[0,np.cos(th),-np.sin(th)],[0,np.sin(th),np.cos(th)]], float)
    if axis == "y":
        return np.array([[ np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]], float)
    if axis == "z":
        return np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]], float)
    raise ValueError("axis must be x/y/z")

def global_RT():
    if ROT_AXIS == "custom":
        Rg = RG_CUSTOM
    else:
        Rg = rodrigues(ROT_AXIS, ROT_DEG)
    tg = TG.astype(float)
    return Rg, tg

def bev_RT():
    if not APPLY_BEV:
        return np.eye(3), np.zeros(3)
    if BEV_ROT_AXIS == "custom":
        Rb = RB_CUSTOM
    else:
        Rb = rodrigues(BEV_ROT_AXIS, BEV_ROT_DEG)
    tb = TB.astype(float)
    return Rb, tb

def read_trajectory(txt_path):
    # trajectory format: ts x y z qx qy qz qw
    # returns Nx3 pos, Nx4 quat (wxyz)
    pos, quat_wxyz = [], []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            toks = re.split(r"[,\s]+", line)
            if len(toks) >= 8:
                _, x, y, z, qx, qy, qz, qw = toks[:8]
            elif len(toks) == 7:  # no timestamp
                x, y, z, qx, qy, qz, qw = toks
            else:
                continue
            x, y, z, qx, qy, qz, qw = map(float, (x,y,z,qx,qy,qz,qw))
            pos.append([x,y,z])
            quat_wxyz.append([qw,qx,qy,qz])  # xyzw -> wxyz
    if not pos:
        raise ValueError("No valid poses parsed from trajectory file.")
    return np.asarray(pos, float), np.asarray(quat_wxyz, float)

def quat_wxyz_to_R(q):
    # convert quaternion (camera-to-world) to rotation matrix
    w,x,y,z = q
    n = w*w + x*x + y*y + z*z
    if n < 1e-12: return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return np.array([
        [1-(yy+zz),   xy-wz,     xz+wy],
        [xy+wz,       1-(xx+zz), yz-wx],
        [xz-wy,       yz+wx,     1-(xx+yy)]
    ], float)

def synthetic_intrinsics_from_hfov(width, height, hfov_deg):
    hfov = np.deg2rad(hfov_deg)
    fx = 0.5 * width / np.tan(0.5 * hfov)
    vfov = 2.0 * np.arctan((height/width) * np.tan(0.5 * hfov))
    fy = 0.5 * height / np.tan(0.5 * vfov)
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    return fx, fy, cx, cy

def make_frustum_corners_cam(fx, fy, cx, cy, w, h, depth):
    Kinv = np.array([[1.0/fx, 0.0,   -cx/fx],
                     [0.0,    1.0/fy,-cy/fy],
                     [0.0,    0.0,    1.0 ]], float)
    corners_px = np.array([[0,   0,   1],
                           [w-1, 0,   1],
                           [w-1, h-1, 1],
                           [0,   h-1, 1]], float).T
    rays = Kinv @ corners_px
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)
    corners_cam = (rays * depth).T
    origin = np.zeros((1,3), float)
    return np.vstack([origin, corners_cam])

def build_frusta_lineset(poses, quats_wxyz, fx, fy, cx, cy, w, h, depth, every, color, Rtot, ttot):
    # batch all frusta as one LineSet. Poses are C2W. Apply combined (Rtot,ttot).
    frustum_local = make_frustum_corners_cam(fx, fy, cx, cy, w, h, depth)
    base_edges = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]], int)
    all_pts, all_lines, all_cols = [], [], []
    idx = 0
    for i in range(0, len(poses), every):
        t = poses[i]
        R_c2w = quat_wxyz_to_R(quats_wxyz[i])  # C2W
        fr_world = (R_c2w @ frustum_local.T).T + t         # world
        fr_world = (Rtot @ fr_world.T).T + ttot            # combined
        all_pts.append(fr_world)
        all_lines.append(base_edges + idx)
        all_cols.append(np.tile(np.asarray(color)[None,:], (len(base_edges), 1)))
        idx += 5
    P = np.vstack(all_pts) if all_pts else np.zeros((0,3))
    L = np.vstack(all_lines) if all_lines else np.zeros((0,2), int)
    C = np.vstack(all_cols) if all_cols else np.zeros((0,3))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(P)
    ls.lines  = o3d.utility.Vector2iVector(L)
    ls.colors = o3d.utility.Vector3dVector(C)
    return ls

def build_trajectory_lines(poses, color=(0,1,0), every=1, Rtot=None, ttot=None):
    # connect pose centers with lines, applying combined transform
    pts = poses[::every]
    if pts.shape[0] < 2:
        return None
    if Rtot is not None and ttot is not None:
        pts = (Rtot @ pts.T).T + ttot
    lines = np.array([[i, i+1] for i in range(len(pts)-1)], int)
    colors = np.tile(np.asarray(color)[None,:], (len(lines), 1))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def crop_top_along_z(points, colors, crop_cm):
    # remove top crop_cm along +Z (in current frame)
    if crop_cm <= 0:
        return points, colors
    crop_m = crop_cm / 100.0
    z = points[:, 2]
    z_max = np.max(z)
    keep = z < (z_max - crop_m)
    points = points[keep]
    if colors is not None:
        colors = colors[keep]
    return points, colors

def signal_handler(sig, frame):
    # Ctrl+C / Exit
    print("\n[INFO] Interrupted by user. Closing window...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    ply_path = Path(PLY_PATH)
    traj_path = Path(TRAJ_PATH)
    if not ply_path.exists():
        print(f"[ERROR] PLY not found: {ply_path}"); sys.exit(1)
    if not traj_path.exists():
        print(f"[ERROR] Trajectory not found: {traj_path}"); sys.exit(1)

    # Compose transforms: FIRST global (Rg,tg), THEN BEV (Rb,tb)
    Rg, tg = global_RT()
    Rb, tb = bev_RT()

    # Combined transform X' = (Rb*Rg) X + (Rb*tg + tb)
    Rtot = Rb @ Rg
    ttot = Rb @ tg + tb

    # Load point cloud
    cloud = o3d.io.read_point_cloud(str(ply_path))
    if cloud.is_empty():
        print("[ERROR] Loaded PLY is empty."); sys.exit(1)

    # Apply combined transform and crop ceilings
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None

    points = (Rtot @ points.T).T + ttot
    points, colors = crop_top_along_z(points, colors, CROP_TOP_CM)

    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)

    # Load trajectory
    poses, quats_wxyz = read_trajectory(str(traj_path))
    print(f"[INFO] poses loaded: {len(poses)} (drawing every {DRAW_EVERY})")

    # Build frusta and trajectory (both get combined transform)
    fx, fy, cx, cy = synthetic_intrinsics_from_hfov(IMG_W, IMG_H, HFOV_DEG)
    frusta = build_frusta_lineset(
        poses, quats_wxyz, fx, fy, cx, cy,
        IMG_W, IMG_H, FRUSTUM_DEPTH, DRAW_EVERY, FRUSTUM_COLOR, Rtot, ttot
    )
    traj_lines = build_trajectory_lines(poses, color=TRAJ_COLOR, every=DRAW_EVERY, Rtot=Rtot, ttot=ttot)

    # World axis
    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=WORLD_AXIS_SIZE)
    W = np.eye(4); W[:3,:3] = Rtot; W[:3,3] = ttot
    world_axis.transform(W)

    geoms = [cloud, frusta, world_axis]
    if traj_lines: geoms.append(traj_lines)

    print("[INFO] Start rendering… (press Esc to exit)")
    try:
        o3d.visualization.draw_geometries(geoms, window_name="Point Cloud + Trajectory (Navigation reloc)")
    except KeyboardInterrupt:
        print("\n[INFO] Window closed by keyboard interrupt.")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Visualization error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
