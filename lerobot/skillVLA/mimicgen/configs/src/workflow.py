"""Read-only planning for MimicGen workflows; no scheduling or GPU work."""
import argparse
import json
import subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('--config',required=True);a=p.parse_args()
 root=next(x for x in Path(__file__).resolve().parents if (x/'lerobot/src/lerobot').is_dir())
 shared=root/'lerobot/skillVLA/libero/configs'
 if a.kind in {'dp','sbd','fsq'}:
  resolver=shared/'train_skills/src/train_skills_config.py'
  extra=['--stage',{'dp':'dp','sbd':'build_data','fsq':'all'}[a.kind]]
 elif a.kind=='pt_vsa':resolver=shared/'train_skillVLA/PT/src/stage1_train_config.py';extra=[]
 elif a.kind=='ft_vsa':resolver=shared/'train_skillVLA/FT/VSA/src/newtask_ft_vsa_config.py';extra=[]
 else:
  comp=a.kind.split('_',1)[1]
  if a.kind.startswith('pt_'):resolver=shared/'train_skillVLA/PT/Terminator/src/terminator_train_config.py';extra=[]
  else:resolver=shared/'train_skillVLA/PT/Terminator/src/terminator_train_config.py';extra=[]
 subprocess.run([str(root/'.venv/bin/python'),str(resolver),'--config',a.config,*extra],check=True)
if __name__=='__main__': main()
