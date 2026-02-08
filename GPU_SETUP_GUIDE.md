# GPU Setup Guide for TrafficRL

## 🎯 Quick GPU Check

Before training, **always** verify GPU availability:

```bash
python scripts\check_gpu.py
```

This will show:
- ✓ GPU name and CUDA version
- ✓ Available GPU memory
- ✓ GPU computation test results

---

## ✅ Confirming GPU Usage

All training scripts automatically detect and use GPU if available:

### 1. **Existing Single-Intersection Training**
```bash
python scripts\train_dqn.py --config traffic_rl\config\config.yaml
```
**GPU Check:** Look for this output at startup:
```
GPU detected: NVIDIA GeForce RTX 3060
CUDA version: 11.8
Using device: cuda:0
```

### 2. **New Multi-Intersection Grid Training**
```bash
python scripts\train_multihead_grid.py --episodes 100
```
**GPU Check:** Look for this output:
```
============================================================
GPU/CUDA Setup
============================================================
✓ GPU detected: NVIDIA GeForce RTX 3060
✓ CUDA version: 11.8
✓ Number of GPUs: 1
✓ Current GPU memory: 12.00 GB
✓ cuDNN auto-tuner enabled
✓ Using device: cuda:0
============================================================
```

---

## ⚠️ If GPU is NOT Detected

You'll see this warning:
```
⚠️  WARNING: CUDA not available!
⚠️  Training will run on CPU (this will be VERY slow)
⚠️  Please install CUDA-enabled PyTorch for GPU acceleration
```

### Fix: Install CUDA-Enabled PyTorch

1. **Check your CUDA version:**
```bash
nvidia-smi
```

2. **Install PyTorch with CUDA support:**

For **CUDA 11.8** (most common):
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For **CUDA 12.1**:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For **CPU only** (not recommended):
```bash
pip install torch torchvision torchaudio
```

3. **Verify installation:**
```bash
python scripts\check_gpu.py
```

---

## 🚀 Performance Comparison

| Configuration | Training Speed | 100 Episodes |
|--------------|----------------|--------------|
| **GPU (RTX 3060)** | ~30 sec/episode | ~50 minutes |
| **CPU (Intel i7)** | ~10 min/episode | ~16 hours |

**GPU is ~20x faster!**

---

## 💡 GPU Optimization Tips

### 1. **Monitor GPU Usage During Training**

Open a new terminal and run:
```bash
# Windows
nvidia-smi -l 1

# This updates every 1 second
```

You should see:
- GPU utilization: 70-95%
- Memory usage: Increasing during training
- Temperature: 60-80°C (normal)

### 2. **Increase Batch Size for Better GPU Utilization**

If GPU utilization is low (<50%), increase batch size in config:

```yaml
# traffic_rl/config/config.yaml
dqn:
  batch_size: 128  # Increase from 64
```

### 3. **Enable cuDNN Auto-Tuner** (Already enabled in grid training)

This is automatically enabled in the new grid training script:
```python
torch.backends.cudnn.benchmark = True
```

---

## 🔍 Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1:** Reduce batch size
```yaml
dqn:
  batch_size: 32  # Reduce from 64
```

**Solution 2:** Reduce buffer size
```yaml
dqn:
  buffer_size: 25000  # Reduce from 50000
```

### Issue: GPU not being used even though CUDA is available

**Check:**
1. Verify PyTorch can see GPU:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print: `True`

2. Check if models are on GPU:
   ```bash
   python scripts\check_gpu.py
   ```

---

## 📊 Expected GPU Memory Usage

| Training Type | GPU Memory | Recommended GPU |
|--------------|------------|-----------------|
| Single Intersection | ~2-3 GB | GTX 1060 (6GB) or better |
| 3×3 Grid (9 agents) | ~4-6 GB | RTX 3060 (12GB) or better |
| 5×5 Grid (25 agents) | ~8-12 GB | RTX 3080 (16GB) or better |

---

## ✅ Final Checklist

Before starting training:

- [ ] Run `python scripts\check_gpu.py` - Should show GPU detected
- [ ] Check `nvidia-smi` - GPU should be visible
- [ ] Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"` - Should print `True`
- [ ] Start training and verify GPU usage in first few seconds

**If all checks pass, you're ready for fast GPU-accelerated training! 🚀**
