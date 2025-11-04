# Robust Multi-Camera Tracking in Terminal Environments 👋

> **Official Implementation of the Manuscript**  
> **"Robust Multi-Camera Tracking in Terminal Environments: A Spatio-Temporal-Appearance Fusion Approach"**  
> *Submitted to The Visual Computer*  
> **Authors:** Dang Wanli, Cheng Jian, Luo Qian, Zheng Huaiyu  
> 
> **If you use this code in your research, please consider citing our paper.**
> *Citation format will be updated upon publication.*

---

## 🚀 Quick Start

### Prerequisites
- **Python** 3.8+
- **PyTorch** 1.12.0+
- **OpenCV** 4.5.0+
- **Ubuntu** 20.04 LTS (recommended)

### Installation
```bash
# Clone this repository
git clone https://github.com/DangWanli0417/mtmct-airport-terminal.git
cd mtmct-airport-terminal

# Install dependencies
pip install -r requirements.txt
```
## 🏗️ System Architecture
```bash
- **scr/MainProcess.py**  Main orchestrator  Multi-process scheduling, camera task distribution
- **scr/FrameDispatcher.py**  Video frame distributor	  Handles local video files for testing
- **scr/DetectLauncher.py**  Pedestrian detector	  Customizable detection models
- **scr/PersonFeatureExtractorLauncher.py**  Feature extractor	  Enhanced with color autocorrelation analysis
- **scr/ReIdLauncher.py**  Multi-camera tracker	  Single & cross-camera tracking
- **scr/QueryFrameDispatcher.py**  Query image handler	    Processes target person images for retrieval
- **scr/HistoryTrackDataBaseCenter.py**  Trajectory database	 In-memory storage & cross-camera matching




