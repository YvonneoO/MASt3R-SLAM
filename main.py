import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
import numpy as np
import os, re, glob

def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    # [map_relocalize]: index the loaded keyframes into the retrieval database
    if len(keyframes) > 0:
        for idx in range(len(keyframes)):
            retrieval_database.update(
                keyframes[idx],
                add_after_query=True,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
        print(f"[backend] retrieval DB primed with {len(keyframes)} prebuilt keyframes")

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)

# ========= Helper functions to load prebuilt map into SharedKeyframes =========
def _parse_traj_txt(path):
    # traj txt format: ts x y z qx qy qz qw
    # returns list[dict]: {"ts": float, "t": float32[3], "q": float32[4] (wxyz)}
    entries = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%":
                continue
            toks = re.split(r"[,\s]+", s)
            if len(toks) < 8:
                continue
            ts, x, y, z, qx, qy, qz, qw = toks[:8]
            entries.append({
                "ts": float(ts),
                "t":  np.array([float(x), float(y), float(z)], dtype=np.float32),
                "q":  np.array([float(qw), float(qx), float(qy), float(qz)], dtype=np.float32),
            })
    if not entries:
        raise ValueError(f"No valid poses parsed from {path}")
    return entries

def _filename_ts(p):
    # extract timestamp from the keyframe img filename
    base = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r"(\d+(?:[._]\d+)?)", base)
    if not m:
        return None
    return float(m.group(1).replace("_", ".").replace(",", "."))

def _quat_wxyz_to_R(qwxyz):
    w, x, y, z = qwxyz
    n = w*w + x*x + y*y + z*z
    if n < 1e-12: 
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    R = np.array([
        [1-(yy+zz),   xy-wz,     xz+wy],
        [xy+wz,       1-(xx+zz), yz-wx],
        [xz-wy,       yz+wx,     1-(xx+yy)]
    ], dtype=np.float32)
    return R

def _pose_to_se3_tensor(t_xyz, q_wxyz, device):
    # Convert cam2world pose to lietorch.Sim3
    # Convert quaternion from wxyz to xyzw format for lietorch
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)

    # Create Sim3 parameters: [t_x, t_y, t_z, q_x, q_y, q_z, q_w, s]
    sim3_params = np.concatenate([t_xyz, q_xyzw, [1.0]], dtype=np.float32)

    return lietorch.Sim3(torch.from_numpy(sim3_params).to(device).unsqueeze(0))

def load_prebuilt_map(model, keyframes, device, cfg, prebuilt_kf_dir, prebuilt_traj_path):
    # Rebuild keyframes (pointmaps) from saved kf imgs and traj,
    # insert into SharedKeyframes,
    # rebuilt retrieval database in backend

    # Load trajectory entries
    entries = _parse_traj_txt(prebuilt_traj_path)

    # Collect kf imgs
    img_paths = sorted(
        glob.glob(os.path.join(prebuilt_kf_dir, "*.png")) +
        glob.glob(os.path.join(prebuilt_kf_dir, "*.jpg")) +
        glob.glob(os.path.join(prebuilt_kf_dir, "*.jpeg"))
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {prebuilt_kf_dir}")

    # pair by timestamps in img filenames
    file_ts = [_filename_ts(p) for p in img_paths]
    use_ts  = all(ts is not None for ts in file_ts)

    load = 0
    for i, p in enumerate(img_paths):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # match the corresponding pose by timestamp
        if use_ts:
            ts = file_ts[i]
            jbest, dtbest = 0, 1e18
            for j, e in enumerate(entries):
                dt = abs(e["ts"] - ts)
                if dt < dtbest:
                    jbest, dtbest = j, dt
            e = entries[jbest]
        else:
            # otherwise match by index order
            j = min(i, len(entries)-1)
            e = entries[j]

        T_c2w = _pose_to_se3_tensor(e["t"], e["q"], device)

        H, W = rgb.shape[:2]
        frame = create_frame(i, rgb, T_c2w, img_size=512, device=device)

        X, C = mast3r_inference_mono(model, frame)
        frame.update_pointmap(X, C)

        keyframes.append(frame)
        load += 1

    print(f"[load_prebuilt_map] Inserted {load} keyframes from {prebuilt_kf_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    # [map_relocalize]: parse prebuilt map args
    parser.add_argument("--prebuilt_kf_dir", default="", help="Path to saved keyframes (RGB).")
    parser.add_argument("--prebuilt_traj", default="", help="Path to trajectory txt of the prebuilt map (ts x y z qx qy qz qw).")
    parser.add_argument("--map_relocalize", action="store_true",
                        help="Relocalize new frames against prebuilt map; do not add new keyframes/landmarks.")

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w, 128)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    # parse args for map_relocalize mode
    config["map_relocalize"]     = bool(args.map_relocalize)
    config["prebuilt_kf_dir"]   = args.prebuilt_kf_dir
    config["prebuilt_traj"]     = args.prebuilt_traj
    # load the prebuilt map into SharedKeyframes
    prebuilt_kf_count = 0
    if args.prebuilt_kf_dir and args.prebuilt_traj:
        load_prebuilt_map(model, keyframes, device, config, args.prebuilt_kf_dir, args.prebuilt_traj)
        prebuilt_kf_count = len(keyframes)
        print(f"[map_relocalize] Loaded {prebuilt_kf_count} memory keyframes")


    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # [map_relocalize] init with "relocalization" mode
            if config.get("map_relocalize", False) and len(keyframes) > 0:
                X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                states.set_frame(frame)
                states.set_mode(Mode.RELOC)
                states.queue_reloc()
                i += 1
                continue
            else:
                # Original: Initialize via mono inference, and encoded features neeed for database
                X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
                states.set_mode(Mode.TRACKING)
                states.set_frame(frame)
                i += 1
                continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if config.get("map_relocalize", False):
            # do not add new keyframes from tracking mode
            add_new_kf = False

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}_reloc.txt", dataset.timestamps, keyframes, start_idx=prebuilt_kf_count)

        # [map_relocalize] save the reconstruction with the loaded scene
        if config.get("map_relocalize", False):
            print(f"[map_relocalize] Saving reconstruction with {len(keyframes)} total keyframes (including prebuilt scene)")
            # [map_relocalize] use new ply save function that preserves the loaded scene
            eval.save_relocalize_reconstruction(
                save_dir,
                f"{seq_name}_reloc.ply",
                keyframes,
                last_msg.C_conf_threshold,
                prebuilt_kf_count,
            )
        else:
            eval.save_reconstruction(
                save_dir,
                f"{seq_name}.ply",
                keyframes,
                last_msg.C_conf_threshold,
            )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
