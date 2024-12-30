
![agibot_world](https://github.com/user-attachments/assets/df64b543-db82-41ee-adda-799970e8a198)



<div id="top" align="center">

[![Static Badge](https://img.shields.io/badge/Download-grey?style=plastic&logo=huggingface&logoColor=yellow)](https://huggingface.co/agibot-world) [![Static Badge](https://img.shields.io/badge/Project%20Page-blue?style=plastic)](https://agibot-world.com) ![Static Badge](https://img.shields.io/badge/License-MIT-blue?style=plastic)

</div>

## Key Features 🔑 <a name="keyfeatures"></a>
- **One million+** trajectories from 100 robots.
- **100+ real-world scenarios** across 5 target domains.
- **Cutting-edge hardware:** visual tactile sensors / 6-DoF Dexterous hand / mobile dual-arm robots
- **Tasks involving:**
    - Contact-rich manipulation
    - Long-horizon planning
    - Multi-robot collaboration


## News <a name="news"></a>

- **`[2024/12/30]`** 🤖 Agibot World demo released.

## Table of Contents
1. [Key Features](#keyfeatures)
2. [Getting Started](#gettingstarted)  
   - [Prepare the AgiBot World data](#preaparedata)
   - [Dataset Preprocessing](#datasetpreprocessing)
   - [Training](#training)
3. [TODO List](#todolist)
4. [License and Citation](#liscenseandcitation)

## Getting started 🔥 <a name="gettingstarted"></a>

#### Prepare the AgiBot World data <a name="preaparedata"></a>

```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
```

#### Dataset Preprocessing <a name="datasetpreprocessing"></a>

Our project relies solely on the [lerobot library](https://github.com/huggingface/lerobot) (dataset `v2.0`), please follow their [installation instructions](https://github.com/huggingface/lerobot?tab=readme-ov-file#installation).
Here, we provide scripts for converting it to the lerobot format.
```
python scripts/convert_to_lerobot.py --src_path /path/to/agibotworld/alpha --task_id 352 --tgt_path /path/to/save/lerobot
```
We would like to express our gratitude to the developers of lerobot for their outstanding contributions to the open-source community.

#### Training <a name="training"></a>

To train a simple Diffusion Policy, please refer to this [Jupyter](https://github.com/OpenDriveLab/AgiBot-World/blob/main/AgibotWorld.ipynb).

<p align="right">(<a href="#top">back to top</a>)</p>

## TODO List 📅 <a name="todolist"></a>

- [x] **AgiBot World Alpha**
- [ ] **AgiBot World Beta**: ~1,000,000 trajectories of high-quality robot data coming by the end of Q1 2025.
  - [ ] Complete language annotation 
- [ ] **AgiBot World Colosseum**:Comprehensive platform launching in 2025.
- [ ] **2025 AgiBot World Challenge**

<p align="right">(<a href="#top">back to top</a>)</p>


## License and Citation   <a name="liscenseandcitation"></a>
All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider citing our project if it helps your research.

```BibTeX
@misc{contributors2024agibotworldrepo,
  title={AgiBot World Colosseum},
  author={AgiBot World Colosseum contributors},
  howpublished={\url{https://github.com/OpenDriveLab/AgiBot-World}},
  year={2024}
}
```
