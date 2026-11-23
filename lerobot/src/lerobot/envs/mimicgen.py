"""Coffee Gym adapter using the pinned, separate MimicGen Python environment."""
from __future__ import annotations
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import subprocess
import os
import json
import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np

INSTRUCTION = 'pick up the coffee pod, place it in the machine, and close the lid'


def project_root():
    return next(p for p in Path(__file__).resolve().parents if (p/'setup_env.sh').is_file())


class MimicGenGymEnv(gym.Env):
    metadata = {'render_modes':['rgb_array'],'render_fps':20}

    def __init__(self,cfg,render=True):
        self.cfg=cfg;self.task_description=INSTRUCTION;self.task=cfg.task;self.wrist_camera_mirror_x=False;self.render_mode='rgb_array';self._max_episode_steps=cfg.episode_length or 600
        self.episode_index=cfg.init_state_offset;self.init_state_id=self.episode_index
        self.action_space=spaces.Box(-1.,1.,shape=(7,),dtype=np.float32)
        h,w=cfg.observation_height,cfg.observation_width
        self.observation_space=spaces.Dict({
            'pixels':spaces.Dict({key:spaces.Box(0,255,shape=(h,w,3),dtype=np.uint8) for key in ['image','image2']}),
            'robot_state':spaces.Dict({'eef':spaces.Dict({'pos':spaces.Box(-np.inf,np.inf,(3,),np.float32),
                                                           'quat':spaces.Box(-1.,1.,(4,),np.float32)}),
                                      'gripper':spaces.Dict({'qpos':spaces.Box(-np.inf,np.inf,(2,),np.float32)})})})
        root=project_root();python=Path(cfg.simulator_python) if cfg.simulator_python else root/'.venv_mimicgen/bin/python'
        metadata=Path(cfg.dataset_dir) if cfg.dataset_dir else root/'dataset_mimicgen/coffee_d0_full_full'
        if not metadata.is_absolute():metadata=root/metadata
        self._camera=json.loads((metadata/'meta/mimicgen_camera.json').read_text())
        if not python.is_absolute():python=root/python
        if not python.is_file():raise FileNotFoundError(f'{python}: run bash setup_env.sh --mimicgen-only')
        parent,child=socket.socketpair();fd=child.fileno()
        child_env=os.environ.copy();child_env.pop('PYTHONPATH',None);child_env.setdefault('MUJOCO_GL','egl')
        # This interpreter must retain its venv path; do not resolve bin/python.
        self._process=subprocess.Popen([str(python),str(root/'lerobot/skillVLA/mimicgen/configs/src/simulator/rollout_worker.py'),'--fd',str(fd)],pass_fds=(fd,),env=child_env)
        child.close();self._connection=Connection(parent.detach());self._last=None;self._steps=0
        try:
            self._receive()
            self._rpc('create',dict(metadata=str(metadata/'eval_env_meta/demo.hdf5'),task=cfg.task,render=render,height=h,width=w))
        except Exception:
            self.close();raise

    def _receive(self):
        if not self._connection.poll(180):raise TimeoutError('MimicGen simulator did not respond within 180s')
        kind,result=self._connection.recv()
        if kind!='ok':raise RuntimeError('MimicGen simulator failed:\n'+result)
        return result

    def _rpc(self,op,args=None):
        self._connection.send((op,args or {}));return self._receive()

    def _observation(self,data):
        self._last=data
        s=data['state']
        if 'image' not in data:
            h,w=self.cfg.observation_height,self.cfg.observation_width
            data={**data,'image':np.zeros((h,w,3),dtype=np.uint8),'wrist_image':np.zeros((h,w,3),dtype=np.uint8)}
        return dict(pixels={'image':data['image'],'image2':data['wrist_image']},robot_state={'eef':{'pos':s[:3],'quat':s[3:7]},'gripper':{'qpos':s[7:9]}})

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.init_state_id=self.episode_index
        self.episode_index+=1;self._steps=0
        obs=self._rpc('reset',{'seed':int(seed if seed is not None else self.init_state_id)})
        return self._observation(obs),{'task':INSTRUCTION,'init_state_id':self.init_state_id}

    def step(self,action):
        result=self._rpc('step',{'action':np.asarray(action).tolist()});self._steps+=1
        success=result['success']
        return (self._observation(result['observation']),result['reward'],success and self.cfg.terminate_on_success,
                self._steps>=self._max_episode_steps,{'is_success':success,'task':INSTRUCTION})

    def camera_transform(self,*,height,width):
        """Projection in the shared overlay convention, preserving Coffee pixels."""
        projection=np.asarray(self._camera['top_world_to_pixel'],dtype=np.float64)
        scale=np.eye(4);scale[0,0]=width/self._camera['width'];scale[1,1]=height/self._camera['height']
        mirror=np.eye(4);mirror[0,0]=-1;mirror[0,2]=width-1
        return mirror@scale@projection

    def render(self):
        if self._last is None:raise RuntimeError('Call reset before render')
        return self._last.get('image')

    def close(self):
        process=getattr(self,'_process',None)
        if process is None:return
        connection=getattr(self,'_connection',None)
        if process.poll() is None:
            try:
                connection.send(('close',{}))
                if connection.poll(5):connection.recv()
                process.wait(timeout=5)
            except (OSError,EOFError,subprocess.TimeoutExpired):
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        if connection is not None:connection.close()
        self._process=None


def create_mimicgen_envs(cfg,n_envs,env_cls):
    if cfg.task_ids not in (None, [0]):raise ValueError('Coffee_D0 has one task: task_ids=[0]')
    if cfg.init_states:raise ValueError('Coffee dataset has no exact simulator states; use init_states=false')
    configs=[copy.deepcopy(cfg) for _ in range(n_envs)]
    for index, item in enumerate(configs):item.init_state_offset+=index
    return {cfg.task:{0:env_cls([partial(MimicGenGymEnv,item) for item in configs])}}
