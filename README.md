<div id="top" align="center">

![agibot_world](https://github.com/user-attachments/assets/df64b543-db82-41ee-adda-799970e8a198)


[![Static Badge](https://img.shields.io/badge/Download-grey?style=plastic&logo=huggingface&logoColor=yellow)](https://huggingface.co/agibot-world) [![Static Badge](https://img.shields.io/badge/Project%20Page-blue?style=plastic)](https://agibot-world.com) ![Static Badge](https://img.shields.io/badge/License-MIT-blue?style=plastic)

</div>

## Key Features 🔑 <a name="keyfeatures"></a>

- **1 million+** trajectories from 100 robots.
- **100+ 1:1 replicated real-life scenarios** across 5 target domains.
- **Cutting-edge hardware:** visual tactile sensors / 6-DoF Dexterous hand / mobile dual-arm robots
- **Wide-spectrum versatile challenging tasks**

<div style="max-width: 100%; overflow-x: auto; margin: 0 auto;">
  <table style="border-collapse: collapse; border: none; width: 100%; table-layout: fixed;">
    <tr style="border: none;">
      <td align="center" style="border: none; padding: 10px;">
        <img src="assets/Contact-rich_manipulation.gif" alt="Contact-rich Manipulation" width="230" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <p><b>Contact-rich Manipulation</b></p>
      </td>
      <td align="center" style="border: none; padding: 10px;">
        <img src="assets/Long-horizon_planning.gif" alt="Long-horizon Planning" width="230" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <p><b>Long-horizon Planning</b></p>
      </td>
      <td align="center" style="border: none; padding: 10px;">
        <img src="assets/Multi-robot_collaboration.gif" alt="Multi-robot Collaboration" width="230" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
        <p><b>Multi-robot Collaboration</b></p>
      </td>
    </tr>
  </table>
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

Download data from our [HuggingFace](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) page.

``` your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
```
Convert the data to **LeRobot Dataset** format following the detailed instructions [here](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha#dataset-preprocessing).

#### Policy Training Quickstart <a name="training"></a>

Leveraging the simplicity of [LeRobot Dataset](https://github.com/huggingface/lerobot), we provide a user-friendly [Jupyter Notebook](https://github.com/OpenDriveLab/AgiBot-World/blob/main/AgibotWorld.ipynb) for training diffusion policy on AgiBot World Dataset.

<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->

## TODO List 📅 <a name="todolist"></a>

- [x] **AgiBot World Alpha**
- [ ] **AgiBot World Beta** (expected Q1 2025)
  - [ ] ~1,000,000 trajectories of high-quality robot data 
  - [ ] ACT、DP3、OpenVLA and some other baseline models
- [ ] **AgiBot World Colosseum** (expected 2025)
  - [ ] A comprehensive platform with toolkits including teleoperation, training and inference.
- [ ] **2025 AgiBot World Challenge** (expected 2025)

<!-- <p align="right">(<a href="#top">back to top</a>)</p> -->


## License and Citation📄   <a name="liscenseandcitation"></a>

All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider citing our project if it helps your research.

```BibTeX
@misc{contributors2024agibotworldrepo,
  title={AgiBot World Colosseum},
  author={AgiBot World Colosseum contributors},
  howpublished={\url{https://github.com/OpenDriveLab/AgiBot-World}},
  year={2024}
}
```