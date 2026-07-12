#!/usr/bin/env bash
# node200 죽은 GPU 슬롯 점유용 blocker.
# 배경: node200은 GPU 3장이 버스에서 빠졌는데 Slurm gres는 여전히 8장 — 죽은 슬롯이 영원히 "빈 자리"로
# 보여 새 잡에 우선 배정되고, 그 잡은 CUDA init 실패 → 조용히 CPU 폴백(학습 진행 0)이 된다.
# 동작: 이 잡은 자기 슬롯의 CUDA를 실제로 초기화해보고
#   - 산 GPU면  → 즉시 종료 (자리 반납)
#   - 죽은 슬롯이면 → walltime까지 sleep (자리를 깔고 앉음)
# 사용법: 8개 제출 → 1분 뒤 살아남은(RUNNING) blocker들이 죽은 슬롯을 전부 점유 → 그 다음 진짜 학습
# 잡을 node200으로 제출하면 산 GPU만 받는다.
#   for i in $(seq 8); do sbatch tools/slurm/node200_gpu_blocker.sbatch; done
#   squeue -u $USER -n n200blk        # 남은 것 = 죽은 슬롯 개수(예상 3)
#   ... (학습 제출) ...
#   scancel -u $USER -n n200blk       # 노드 수리되면/다 쓰면 반드시 정리
#SBATCH --job-name=n200blk
#SBATCH --partition=debug
#SBATCH --qos=base_qos
#SBATCH --nodelist=node200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data2/dohyeon/SBD/tools/slurm/logs/%x_%j.out
#SBATCH --error=/data2/dohyeon/SBD/tools/slurm/logs/%x_%j.out
set -u
echo "node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
if timeout 120 /data2/dohyeon/SBD/.venv/bin/python -c \
     "import torch; assert torch.cuda.is_available(), 'no cuda'; torch.zeros(8, device='cuda'); print('cuda OK:', torch.cuda.get_device_name(0))"; then
  echo "live GPU slot → exit (자리 반납)"
  exit 0
fi
echo "DEAD GPU slot (CVD=${CUDA_VISIBLE_DEVICES:-}) → walltime까지 점유 (진짜 잡이 이 슬롯을 안 받게)"
sleep infinity
