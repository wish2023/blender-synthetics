# Blender Synthetics

Create synthetic datasets for object detection using Blender.

## Description

Blender Synthetics is a repository designed to help you create synthetic datasets for object detection tasks. It automates the process of generating random scenes in Blender, importing custom 3D models with randomized configurations, and rendering numerous images with annotations. These datasets can be used to train and evaluate object detection models.

## Table of Contents

1. [Getting Started](#getting-started)
   - [Clone the Repository](#clone-the-repository)
   - [Linux Installation](#linux-installation)
   - [Windows Installation (with Anaconda)](#windows-installation-with-anaconda)
1. [Generate Synthetics](#generate-synthetics)
1. [Models](#models)
1. [Parameters](#parameters)

## Getting Started

This repo was tested on the following:

- Python 3.10
- Blender 3.2 or 3.3 (Linux), Blender 3.6.4 (Windows)

### Clone the Repository

```bash
git clone https://github.com/tehwenyi/blender-synthetics.git
```

### Linux Installation

1. **Install Blender:**
   - Follow the installation instructions [here](https://www.blender.org/download/).
   - Add Blender to your PATH by running the following command:

        ```bash
        echo 'export PATH=/path/to/blender/directory:$PATH' >> ~/.bashrc
        ```
    
1. **Verify GPU Compatibility:**

    Ensure your GPU is supported by Blender. Refer to the [official documentation](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html) for compatibility details.

1. **Install Required Packages:**

    ```bash
    sh install_requirements.sh
    ```

### Windows Installation (with Anaconda)

1. **Install Blender:**
    
    Download Blender from the official website: [Blender Downloads](https://www.blender.org/download/).

2. **(Anaconda) Create a Virtual Environment with Anaconda:**

    - Follow Anaconda installation instructions: [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/windows/).
    - Launch Anaconda Prompt and create a virtual environment for Blender:

        ```bash
        conda create --name blender python=3.10
        ```

3. **(Anaconda) Add Blender to the PATH:**

    - Create a batch script (a .bat file) that will launch Blender. You can do this using a text editor. Create a new file with a `.bat extension` (e.g., blender_launcher.bat) and add the following line (replace the path with the actual Blender executable path.):

        ```bat
        "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
        ```

    - Copy the batch script to your Anaconda environment's `Scripts` directory. (eg. `"C:\Anaconda3\envs\your_env_name\Scripts"`)
    - If you are not using Anaconda, you will have to do the equivalent of this for your virtual environment/PowerShell.

4. **(Anaconda) Activate Your Anaconda Environment:**

    ```bash
    conda activate blender
    ```

5. **Adding Blender's Executable to the System PATH**

   **Locate the Blender Executable Path:**

   First, find the exact path where Blender is installed. The default installation path for Blender is usually:

   `C:\Program Files\Blender Foundation\Blender <version>\`

   For example:

   `C:\Program Files\Blender Foundation\Blender 3.6\`

    **Add the Blender Path to the System PATH:**

    1. **Open System Properties:**

    - Press `Win + R` to open the Run dialog.
    - Type `sysdm.cpl` and press Enter to open the System Properties window.

    2. **Open Environment Variables:**

    - In the System Properties window, go to the **Advanced** tab.
    - Click on **Environment Variables...** at the bottom.

    3. **Edit the PATH Variable:**

    - In the Environment Variables window, find the `Path` variable under "System variables" or "User variables" (depending on whether you want the change to apply system-wide or just for your user account).
    - Select `Path` and click **Edit...**.

    4. **Add a New Entry:**

    - Click **New** and add the path to your Blender installation (e.g., `C:\Program Files\Blender Foundation\Blender 3.6\`).

    5. **Save and Close:**

    - Click **OK** to close each window and save your changes.

    6. **Restart Command Prompt or PowerShell:**

    - Close and reopen any Command Prompt or PowerShell windows you have open to apply the changes to the PATH variable.

    7. **Test the `blender` Command:**

    - Now, open a new Command Prompt or PowerShell window and type:

    ```bash
    blender
    ```

    - If Blender launches, then it is correctly recognized.

6. **Install bpycv in Blender:**

    You may run the following commands in PowerShell or your virtual environment.

    ```bash
    blender -b --python-expr "from subprocess import sys, call; call([sys.executable,'-m','ensurepip'])"
    blender -b --python-expr "from subprocess import sys, call; call([sys.executable]+'-m pip install --target="$TARGET" -U pip setuptools wheel'.split())"
    blender -b --python-expr "from subprocess import sys, call; call([sys.executable]+'-m pip install --target="$TARGET" -U bpycv'.split())"
    ```

    If the above commands do not work (or bpycv is still not installed), use Blender's scripting terminal:

    1. Open Blender.
    2. Go to the Scripting workspace/tab.
    3. Copy and paste the following lines into the scripting terminal:
    ```bash
    from subprocess import sys, call; call([sys.executable,'-m','ensurepip'])
    from subprocess import sys, call; call([sys.executable]+'-m pip install --target="$TARGET" -U pip setuptools wheel'.split())
    from subprocess import sys, call; call([sys.executable]+'-m pip install --target="$TARGET" -U bpycv'.split())
    ```
    4. Run the script to install bpycv in Blender.
    

7. **Install Required Python Packages in Blender:**

    - Locate Blender's Python executable in the Blender installation directory. (eg. `"C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin\python.exe"`)
    - Ensure you are in the folder of this repo (containing `requirements.txt`).
    - Option 1: Run the following in PowerShell (with administrative rights):
        ```
        & "C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin\python.exe" -m pip install -r requirements.txt
        ```
    - (Anaconda) Option 2: Run the following command in your Anaconda Prompt:
        ```bash
        "C:\Program Files\Blender Foundation\Blender 3.6\3.6\python\bin\python.exe" -m pip install -r requirements.txt
        ```
    - Option 3: Install the packages via your virtual environment.

8. **Setup Completed**

    You have completed the setup. Proceed to [Generate synthetics](#generate-synthetics) to run the synthetic dataset generation process. 

    Generating Images can be done from PowerShell or your virtual environment, but Generating Annotations must be done in your virtual environment.

## Generate Synthetics

To generate synthetic datasets for object detection:

1. **Update `config/models.yaml` and `config/render_parameters.yaml` as needed.** Refer to [Models](#models) and [Parameters](#parameters) for more details.

2. **Generate Images:**

    `blender -b -P src/render_blender.py`

    * Note you have to do this in Anaconda Prompt if on Windows

3. **Generate Annotations:**

    `python3 src/create_labels.py`

## Models
Currently supports fbx/obj/blend. Ensure your models only contain one object that has the same name as its filename.

### Classes

Your targets of interest. Bounding boxes will be drawn around these objects.

### Obstacles (Optional)

Other objects which will be present in the scene. These won't be annotated.

### Scenes (Optional)

Textures that your scene may have. Explore possible textures from [texture haven](https://polyhaven.com/textures) and store all texture subfolders in a main folder.

## Parameters

### Occlusion awareness

When not occlusion aware, bounding boxes will surround regions of the object that aren't visible by the camera.

![occ diagram](diagrams/occlusion.jpg)

#### Visibility threshold

The fraction of an object that must be visible by the camera for it to be considered visible to a human annotator.

![camera diagram](diagrams/visthresh.png)


#### Component visibility threshold

The fraction of an object components that must be visible by the camera for it to be considered visible to a human annotator.

![camera diagram](diagrams/comvisthresh.png)

#### Minimum Pixels

The minimum amount of pixels of an object that must be visible by the camera for it to be considered visible to a human annotator. 

### Camera configurations

![camera diagram](diagrams/camera.png)

### Sun

The sun's [energy](https://docs.blender.org/manual/en/latest/render/lights/light_object.html) is the light intensity on the scene. The tilt is responsible for casting shadows and works similar to the camera's tilt.
