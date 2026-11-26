
```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=/home/eugene/git/lerobot/downloads/lerobot-pick-wood-from-slots-act-step048845 \
  --device=xpu \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=so101_follower \
  --robot.cameras="{ overview: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0, width: 640, height: 480, fps: 30}}" \
  --task="pick up the blocks from the slots and place them in the middle slot" \
  --fps=30 \
  --duration=60
```