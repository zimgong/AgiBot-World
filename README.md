![agibot_world](https://github.com/user-attachments/assets/df64b543-db82-41ee-adda-799970e8a198)



<div id="top" align="center">


[![Static Badge](https://img.shields.io/badge/Download-grey?style=plastic&logo=huggingface&logoColor=yellow)](https://huggingface.co/agibot-world) [![Static Badge](https://img.shields.io/badge/Project%20Page-blue?style=plastic)](https://agibot-world.com) ![Static Badge](https://img.shields.io/badge/License-MIT-blue?style=plastic)

</div>

## Key Features 🔑 <a name="keyfeatures"></a>

- **1 million+** trajectories from 100 robots.
- **100+ real-world scenarios** across 5 target domains.
- **Cutting-edge hardware:** visual tactile sensors / 6-DoF Dexterous hand / mobile dual-arm robots
- **Tasks involving:**
  - Contact-rich manipulation
  - Long-horizon planning
  - Multi-robot collaboration

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
    <video controls autoplay loop muted width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <source src="assets/Contact-rich_manipulation.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <video controls autoplay loop muted width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <source src="assets/Long-horizon_planning.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <video controls autoplay loop muted width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <source src="assets/Multi-robot_collaboration.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>


## News📰 <a name="news"></a>

- **`[2024/12/30]`** 🤖 Agibot World Alpha released.

## Table of Contents

1. [Key Features](#keyfeatures)
2. [Getting Started](#gettingstarted)  
   - [How to Get Started with Our AgiBot World Data](#preaparedata)
   - [Policy Learning Quickstart](#training)
3. [TODO List](#todolist)
4. [License and Citation](#liscenseandcitation)

## Getting started 🔥 <a name="gettingstarted"></a>

#### How to Get Started with Our AgiBot World Data <a name="preaparedata"></a>

Download the data from our [HuggingFace](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) page.

``` your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
```
Convert the data to LeRobot Dataset format following the detailed instructions [here](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha#dataset-preprocessing).

#### Policy Training Quickstart <a name="training"></a>

Leveraging the simplicity of [LeRobot Dataset](https://github.com/huggingface/lerobot), we provide a user-friendly [Jupyter Notebook](https://github.com/OpenDriveLab/AgiBot-World/blob/main/AgibotWorld.ipynb) for training diffusion policy on AgiBot World Dataset.

<p align="right">(<a href="#top">back to top</a>)</p>

## TODO List 📅 <a name="todolist"></a>

- [x] **AgiBot World Alpha**
- [ ] **AgiBot World Beta**: (expected Q1 2025)
  - [ ] ~1,000,000 trajectories of high-quality robot data 
  - [ ] ACT、DP3、OpenVLA and some other baseline models
- [ ] **AgiBot World Colosseum**:Comprehensive platform (expected 2025)
  - [ ] Comprehensive platform including teleoperation, training, inference tools.
- [ ] **2025 AgiBot World Challenge** (expected 2025)

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