"""Isolated robosuite 1.4 runtime for the Torch process (local pipe RPC).

Only the parent process owns this inherited file descriptor. No network server,
no GPU model imports, and no mixing LIBERO's robosuite into the simulator.
"""
from __future__ import annotations
import argparse
import json
from multiprocessing.connection import Connection
from pathlib import Path
import traceback
import sys
import os


def serve(fd):
    conn = Connection(fd)
    env = None
    try:
        import numpy as np
        import h5py
        source = os.environ.get('MIMICGEN_SOURCE_ROOT')
        if not source:
            root = next(p for p in Path(__file__).resolve().parents if (p/'third_party/mimicgen').is_dir())
            source = root/'third_party/mimicgen'
        sys.path.insert(0,str(source))
        import robosuite
        import mimicgen  # registers Coffee_D0 and CoffeePreparation_D0
        from robosuite.utils.transform_utils import mat2quat
        conn.send(('ok',None))
        while True:
            op, args = conn.recv()
            try:
                if op == 'create':
                    with h5py.File(args['metadata'], 'r') as f:
                        description = json.loads(f['data'].attrs['env_args'])
                    name = args['task']
                    if name not in {'Coffee_D0','CoffeePreparation_D0'}: raise ValueError(f'Unsupported environment {name}')
                    kw = description['env_kwargs']
                    kw.update(has_renderer=False,has_offscreen_renderer=args['render'],use_camera_obs=args['render'],
                              camera_names=['agentview','robot0_eye_in_hand'],camera_heights=args['height'],camera_widths=args['width'],
                              ignore_done=True,hard_reset=False)
                    env = robosuite.make(env_name=name, **kw)
                    env.modify_observable('robot0_joint_pos','active',True)
                    render = args['render']
                    result = None
                elif op == 'reset':
                    np.random.seed(args['seed'])
                    obs = env.reset()
                    result = pack(env,obs,render,mat2quat)
                elif op == 'reset_exact':
                    state = np.asarray(args['state'], dtype=np.float64)
                    if state.ndim != 1 or not np.isfinite(state).all():
                        raise ValueError('Exact reset requires a finite flattened simulator state')
                    # The XML includes body poses that are not in MuJoCo's state vector.
                    env.reset_from_xml_string(env.edit_model_xml(args['model_xml']))
                    if state.shape != env.sim.get_state().flatten().shape:
                        raise ValueError('Exact reset state does not match this simulator model')
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    # XML reset constructs OSC at the model's default joints.
                    # MuJoCo's flattened state does not include that controller's
                    # cached pose or nullspace joint target. Rebase both before
                    # the first relative action, as a normal robot reset does.
                    for robot in env.robots:
                        robot.controller.update_initial_joints(
                            env.sim.data.qpos[robot._ref_joint_pos_indexes]
                        )
                    obs = env._get_observations(force_update=True)
                    result = pack(env,obs,render,mat2quat)
                elif op == 'step':
                    action = np.asarray(args['action'],dtype=np.float32)
                    if action.shape != (7,) or not np.isfinite(action).all(): raise ValueError('Expected finite OSC action[7]')
                    obs, reward, done, info = env.step(np.clip(action,-1,1))
                    result = dict(observation=pack(env,obs,render,mat2quat),reward=float(reward),success=bool(env._check_success()))
                elif op == 'snapshot':
                    # Read-only diagnostics; restoration uses action replay so
                    # controller targets and gripper integration are preserved.
                    result = dict(sim_state=env.sim.get_state().flatten().tolist(),
                                  qacc_warmstart=env.sim.data.qacc_warmstart.tolist(),
                                  body_positions=env.sim.data.body_xpos.tolist(),
                                  body_names=list(env.sim.model.body_names),
                                  controller_initial_joints=[r.controller.initial_joint.tolist() for r in env.robots])
                elif op == 'close':
                    conn.send(('ok',None)); break
                else: raise ValueError(f'Unknown RPC operation {op}')
                conn.send(('ok',result))
            except Exception:
                conn.send(('error',traceback.format_exc()))
    except (EOFError, BrokenPipeError): pass
    except Exception:
        try: conn.send(('error',traceback.format_exc()))
        except (EOFError,BrokenPipeError): pass
    finally:
        if env is not None: env.close()
        conn.close()


def pack(env,obs,render,mat2quat):
    import numpy as np
    result = dict(state=np.concatenate([obs['robot0_eef_pos'],obs['robot0_eef_quat'],obs['robot0_gripper_qpos']]).astype(np.float32),
        joints=np.asarray(obs['robot0_joint_pos'],dtype=np.float32))
    if render:
        # Raw robosuite OpenGL rows -> stored Coffee upright OpenCV video.
        result['image'] = np.ascontiguousarray(obs['agentview_image'][::-1])
        result['wrist_image'] = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1])
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fd',type=int,required=True);serve(p.parse_args().fd)
