# ADiCo-IRL — Anki Vector Deployment

Deploys a trained 3-agent ADiCo navigation policy onto physical Anki Vector robots using an overhead webcam and color-based detection. Single Python process, ~10 Hz control loop, no ROS 2.

---

## Hardware

| Item | Details |
|---|---|
| 3× Anki Vector robots | Unique colored paper taped on top |
| 3× colored goal squares | On arena floor, one per robot |
| 1× overhead USB webcam | Logitech C920, 1m × 1m arena |
| Server running wire-pod | `http://10.0.0.28:8080/` |

---

## One-Time Setup

### 1. Firmware & wire-pod

1. Flash Vector firmware: https://wpsetup.keriganc.com/html/main.html
2. Connect Vector to wire-pod: `http://10.0.0.28:8080/`
3. Verify wire-pod is running: `systemctl status wire-pod`

### 2. Conda environment

```bash
source ~/.bashrc
conda create -n vector_ctrl python=3.9 -y
conda activate vector_ctrl
pip install matplotlib numpy opencv-python torch pyyaml pandas
```

### 3. wirepod Vector SDK

```bash
git clone https://github.com/kercre123/wirepod-vector-python-sdk.git
cd wirepod-vector-python-sdk/
pip install -e .
cd ..
```

> **Note:** Do NOT `pip install anki_vector` from PyPI — that is a different, incompatible package. Always use the wirepod git repo above.

### 4. Robot credentials

The SDK reads from `~/.anki_vector/sdk_config.ini`. Two ways to get it:

**Option A — copy an existing config (if you already have it):**
```bash
mkdir -p ~/.anki_vector
cp /path/to/your/sdk_config.ini ~/.anki_vector/sdk_config.ini
```

**Option B — generate it fresh:**
```bash
python3 -m anki_vector.configure
```
Follow the prompts. You need the robot's serial number and the wire-pod server IP.

### 5. Verify connection

```bash
python3 -c "
import anki_vector
r = anki_vector.Robot()
r.connect()
print('Connected:', r.serial)
r.disconnect()
"
```

### 6. Update config.yaml

Open `config.yaml` and fill in the serial numbers for each robot:

```yaml
agents:
  - id: 0
    serial: "00e20117"   # ← your actual serial
  - id: 1
    serial: "00e20118"
  - id: 2
    serial: "00e20119"
```

---

## Calibration (run once per physical setup change)

```bash
conda activate vector_ctrl
cd /path/to/ADiCo-IRL
python calibrate.py
```

Follow the on-screen prompts:
1. Click 4 arena corners (TL → TR → BR → BL)
2. Click each robot paper, then each goal paper
3. Press **Enter** to confirm each color mask, **r** to re-click

Saves `arena_corners.npy` and updates color HSV ranges in `config.yaml`.

---

## Running

### Test vision + policy (no robots needed)
```bash
python main.py --no-robot
```

### Test movement controller only (figure-8 demo)
```bash
# Simulation only (no robot required)
python movement_controller.py --demo

# On real robot
python movement_controller.py --demo --serial 00e20117

# Saves demo_trajectory.png on exit
```

### Full deployment
```bash
python main.py
```

### Flags
| Flag | Effect |
|---|---|
| `--no-robot` | Skip robot connection, print actions to terminal |
| `--no-viz` | Disable OpenCV debug overlay |
| `--log-level DEBUG` | Verbose logging |
| `--config PATH` | Use a different config file |

---

## Project Structure

```
adico_vector/
├── config.yaml              # All tunable parameters (edit serials + colors here)
├── main.py                  # 10 Hz control loop
├── calibrate.py             # One-time color + corner calibration
├── camera.py                # Webcam capture
├── color_detector.py        # HSV blob detection → robot/goal positions
├── observation_builder.py   # Builds 10-dim VMAS observation vectors
├── policy_runner.py         # Loads .pt checkpoint, runs inference
├── movement_controller.py   # Sends wheel commands to real Vectors
├── policy/
│   └── loader.py            # DiCoPolicy network + checkpoint loader
├── checkpoints/             # Place your .pt files here
└── utils/
    ├── perspective.py       # Homography warp
    └── visualization.py     # Live debug overlay
```

---

## Observation Format (10-dim per agent)

```
[x, y, vx, vy, x-gx, y-gy, x-ox0, y-oy0, x-ox1, y-oy1]
```
- Position: arena origin at center, +x right, +y up (metres)
- `x - gx`: agent minus goal (not goal minus agent)
- Other agents sorted by ID

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `sdk_config.ini` not found | Run `python3 -m anki_vector.configure` |
| Robot not detected by color | Re-run `calibrate.py`; check lighting |
| Policy outputs NaN | Check checkpoint path in `config.yaml` |
| Camera not opening | Check `camera_index` in `config.yaml` (try 0, 1, 2) |
| `anki_vector` import error | Make sure you installed from the wirepod git repo, not PyPI |
| wire-pod not running | `systemctl start wire-pod` |


conda activate vector_ctrl
<!-- vector 3 -->
python -m anki_vector.configure -s 00506502 -n Vector-S9G7 -i 10.0.0.148
<!-- Vector 1 -->
python -m anki_vector.configure -s 00a12dbd -n Vector-X9U9 -i 10.0.0.144
<!-- Vector 2 -->
python -m anki_vector.configure -s 00a132eb -n Vector-G7P7 -i 10.0.0.30   
