# HW1 – setting up the environment on your own computer (Windows / Mac / Linux)

You need **Python 3.10 or newer** (https://www.python.org/downloads/ – on Windows tick *"Add python.exe to PATH"* in the installer), about 3 GB of free disk space, and **internet on the first run** (CIFAR-10, ~170 MB, and the pretrained ResNet20 are downloaded once and cached). A GPU is *not* needed – HW1 should run on the CPU. 

## 1. Unzip and open a terminal in the folder
Unzip `HW1.zip` into a new folder (Windows: right-click → *Extract All*). Open a terminal **in that folder**: Windows – in File Explorer type `powershell` in the address bar and press Enter; Mac/Linux – `cd` into the folder.

## 2. Create a virtual environment and install the requirements (once)
**Windows (PowerShell):**
```powershell
python -m venv hw1-env
.\hw1-env\Scripts\Activate.ps1
```
(Command Prompt `cmd`: `hw1-env\Scripts\activate.bat`.) If PowerShell says that running scripts is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again – this only affects the current window.

**Mac / Linux:**
```bash
python3 -m venv hw1-env
source hw1-env/bin/activate
```

The prompt now starts with `(hw1-env)`. Then (all systems):
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
*(Linux only: to avoid downloading the large GPU build of PyTorch, first run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.)*

Check the installation:
```bash
python -c "import torch, torchvision, sklearn, matplotlib; print('OK', torch.__version__)"
```

## 3. Open the notebook
Always work from an **activated** environment (the prompt shows `(hw1-env)`; after a restart activate it again as in step 2) and from the folder that contains the notebook and `helper.py`.

* **JupyterLab:** run `jupyter lab`, open `hw1_vision_attacks_student.ipynb`, kernel *Python 3*.
* **VS Code:** open the folder, open the notebook, click *Select Kernel* (top right) → *Python Environments* → `hw1-env`.

## 4. Exporting the notebook
With the environment activated:
```bash
jupyter nbconvert --to html hw1_vision_attacks_id1_id2_id3.ipynb
```
creates an HTML file with all the outputs, figures and GIFs. 

## Troubleshooting
* `jupyter` / `pip` not found, or `ModuleNotFoundError: torch` → the environment is not activated.
* `ModuleNotFoundError: helper` → `helper.py` must be in the same folder as the notebook, and Jupyter/VS Code must be opened from that folder.
* Download errors (CIFAR-10 or the pretrained model) → check your internet connection / VPN / proxy and run the cell again.
* To start over, delete the `hw1-env` folder and repeat step 2.
