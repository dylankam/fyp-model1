# Automatic Gesture Creation From Text for Humanoid Robots

A pipeline that converts natural-language speech into synchronised, physically-valid arm gestures for humanoid robots. Text is processed by a two-stage LLM chain; the resulting Cartesian keyframes are solved through a PINK inverse-kinematics solver and streamed to the robot (or a MuJoCo visualiser) in sync with synthesised TTS audio.

---

## Architecture Overview

```
User Text
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                     brain-server.py                     │
│                                                         │
│  LLM 1 (Gemini)         LLM 2 (Gemini / OpenAI /       │
│  ─────────────          Fine-tuned Llama-3 via Colab)   │
│  Intent Analyser    ──► Cartesian Keyframe Mapper        │
│  (gesture + hand)       (normalised XYZ + orientation)  │
│         │                          │                    │
│         └──────────────────────────┘                    │
│                        │                               │
│              PINK IK Solver (pinocchio)                 │
│              Joint Angle Trajectory                     │
│                        │                               │
│              gTTS Audio (base64 MP3)                    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Client (nao-client.py / test-client.py / test_in_mujoco.py)
```

**Supported robots:** NAO · Pepper · iCub · ergoCub (and any URDF robot via `generate_mapping.py`)

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| conda | any recent |
| pinocchio | installed via conda-forge (see below) |
| Google Gemini API key | required |
| OpenAI API key | optional |
| Google Colab (free tier) | required for fine-tuned LLM2 only |

> **NAO client only:** `nao-client.py` targets the NAOqi SDK, which requires **Python 2.7**. See [NAO Client Setup](http://doc.aldebaran.com/2-8/dev/python/install_guide.html).

---

## 2. Clone and Install

### 2.1 Clone the repository

```bash
git clone https://github.com/dylankam/fyp-model1.git
cd fyp-model1
```

### 2.2 Create and activate a conda environment

```bash
conda create -n nao_env python=3.11
conda activate nao_env
```

### 2.3 Install pinocchio via conda-forge (must come first)

```bash
conda install -c conda-forge pinocchio
```

### 2.4 Install remaining Python dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Environment Variables (API Keys)

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required — used by brain-server.py for LLM1 (and default LLM2)
GEMINI_API_KEY=AIza...

# Optional — only needed when LLM2_PROVIDER is set to "openai"
OPENAI_API_KEY=sk-...
```

`brain-server.py` uses `python-dotenv` to load the `.env` file automatically at startup — no manual exporting required.

---

## 4. Fine-Tuned Model Weights (LLM2 — optional)

If you want to locally run the fine-tuned Llama-3-8B LoRA adapter rather than using the commercial APIs, model weights are stored here:

> **[LoRA model weights](https://drive.google.com/file/d/1NlK44-fyfXYh0kmS5C1PYuM3oIFwb503/view?usp=drive_link)**

The model runs inside **Google Colab** (free T4 GPU) and is exposed to the local Brain Server via an ngrok tunnel.

### 4.1 Using the pre-trained weights

1. Open [`finetune/fyp_finetune.ipynb`](finetune/fyp_finetune.ipynb) in Google Colab.
2. Navigate to the **"For if you already have Model Weights"** section.
3. Download `lora_model_backup.zip` from the link above.
4. Upload the zip when prompted by the **"Run This Cell"** cell.
5. Add your `NGROK_TOKEN` to Colab Secrets (left sidebar → key icon):
   - Get a free token at [ngrok.com](https://ngrok.com)
6. Run the server cell — Colab will print a public URL ending in `/generate_gesture`.
7. Copy that URL into `brain-server.py`:
   ```python
   COLAB_NGROK_URL = "https://<your-subdomain>.ngrok-free.app/generate_gesture"
   ```

### 4.2 Training your own weights

1. Open [`finetune/fyp_finetune.ipynb`](finetune/fyp_finetune.ipynb) in Colab.
2. Run all cells from the top — training uses the `finetune/dataset.jsonl` and `finetune/eval_dataset.jsonl` files committed to this repo.
3. After training, the adapter is saved to `lora_model/`. Zip and download it:
   ```python
   # Already included in the notebook:
   !zip -r lora_model_backup.zip lora_model
   ```
4. Store locally and upload to Colab next time.

> **Base model evaluation only:** [`finetune/fyp_base.ipynb`](finetune/fyp_base.ipynb) runs the unmodified `llama-3-8B-Instruct` model through the same ngrok server — useful for comparing base vs fine-tuned performance without uploading weights.

---

## 5. Running the Pipeline End-to-End

### 5.1 Start the Brain Server

```bash
# In the repo root with nao_env active
python brain-server.py
```

Expected output:
```
Gemini Client Initialized: Yes
OpenAI Client Initialized: Yes   # or No if key not set
BRAIN SERVER: Starting on port 65432...
```

### 5.2 Choose a client

#### A — Test client (no robot required, plays audio locally)

```bash
python test-client.py
```

```
=== Brain Server Test Client ===
Robot profile : pepper
Server URL    : http://127.0.0.1:65432/process

Enter text to send (Ctrl+C to quit): Welcome to my final year project demonstration.
```

The client prints the trajectory summary for each sentence and plays the TTS audio via pygame.

#### B — MuJoCo visualiser (requires a URDF robot profile)

Edit the `ROBOT_NAME` constant at the top of `test_in_mujoco.py` (e.g. `"icub"`, `"ergocub"`, `"nao"`), then:

```bash
python test_in_mujoco.py
```

A MuJoCo viewer opens, you enter text, and the robot animates in sync with the audio.

#### C — Physical NAO / Pepper robot

See [NAO Client Setup](http://doc.aldebaran.com/2-8/dev/python/install_guide.html) below.

### 5.3 Switching LLM2 providers

In `brain-server.py`, change the `LLM2_PROVIDER` constant in `process_route()`:

```python
LLM2_PROVIDER = "gemini"     # default — Gemini Flash Lite (no Colab needed)
LLM2_PROVIDER = "openai"     # GPT-4 class model (OPENAI_API_KEY required)
LLM2_PROVIDER = "finetuned"  # Fine-tuned Llama-3 via Colab ngrok tunnel
```

---

## 6. NAO Client (Python 2.7)

`nao-client.py` communicates with a physical or simulated NAO/Pepper via the **NAOqi Python SDK**, which is Python 2.7 only.

### Setup

1. Download the **NAOqi Python 2 SDK** from [Aldebaran Support](http://doc.aldebaran.com/2-8/dev/python/install_guide.html).
2. Add it to your `PYTHONPATH`:
   ```bash
   export PYTHONPATH=/path/to/pynaoqi-sdk/lib/python2.7/site-packages:$PYTHONPATH
   ```
3. Install Python 2 dependencies:
   ```bash
   pip2 install requests pygame
   ```
4. Edit the connection constants at the top of `nao-client.py`:
   ```python
   SERVER_URL = "http://<brain-server-ip>:65432/process"
   ROBOT_IP   = "192.168.1.x"   # physical robot IP
   ROBOT_PORT = 9559             # default NAOqi port
   ```
5. Run with Python 2:
   ```bash
   python2 nao-client.py
   ```

> **Note:** The Brain Server itself runs on Python 3. The NAO client is the only Python 2 component.

---

## 7. Adding a New Robot

1. Run the calibration wizard to auto-generate an orientation function:
   ```bash
   python generate_mapping.py
   ```
   Follow the interactive prompts to identify the hand axes in MuJoCo.

2. Add the generated `get_robot_orientation` function and a new profile entry to `robot_profiles.py`.

3. Reference the new profile key (e.g. `"my_robot"`) in your client script or in `brain-server.py` requests.

---

## 8. Repository Structure

```
fyp-model1/
├── brain-server.py          # Flask API server — LLM chain + PINK IK
├── nao-client.py            # NAOqi client for physical NAO/Pepper (Python 2)
├── test-client.py           # Minimal test client, no robot required
├── test_in_mujoco.py        # MuJoCo simulation client
├── robot_profiles.py        # Robot geometry, IK limits, orientation functions
├── generate_mapping.py      # Interactive wizard to calibrate new robots
├── nao_clean.urdf           # Cleaned NAO URDF for PINK IK
├── requirements.txt
├── .env.example
└── finetune/
    ├── fyp_base.ipynb       # Colab — base Llama-3 ngrok server
    ├── fyp_finetune.ipynb   # Colab — fine-tune + ngrok server
    ├── dataset.jsonl        # Training split
    └── eval_dataset.jsonl   # Evaluation split
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: pinocchio` | Run `conda install -c conda-forge pinocchio` |
| `PINK IK Error: Robot profile 'x' not found` | Check the `ROBOT_NAME` / `"robot"` key matches a key in `ROBOT_PROFILES` |
| `Could not connect to Brain Server` | Ensure `brain-server.py` is running and the port (65432) is not firewalled |
| Colab ngrok tunnel URL expired | Re-run the server cell in the notebook; update `COLAB_NGROK_URL` |
| `GEMINI_API_KEY` not set | Check that `.env` exists in the repo root and contains the key |
| Audio plays but robot does not move | Trajectory may be empty — check Brain Server logs for IK errors |
