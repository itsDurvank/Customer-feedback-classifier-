# 🔧 Troubleshooting Guide - Customer Feedback Analyzer

This comprehensive troubleshooting guide helps you resolve common issues and optimize performance.

## 📋 Table of Contents

- [Installation Issues](#installation-issues)
- [Training Problems](#training-problems)
- [Model Loading Errors](#model-loading-errors)
- [Streamlit App Issues](#streamlit-app-issues)
- [Performance Problems](#performance-problems)
- [Memory Issues](#memory-issues)
- [GPU/CUDA Problems](#gpucuda-problems)
- [Data Issues](#data-issues)
- [Deployment Issues](#deployment-issues)
- [Advanced Debugging](#advanced-debugging)

---

## 🚨 Installation Issues

### Issue: "Python not found" or "python is not recognized"

**Symptoms:**

```bash
'python' is not recognized as an internal or external command
```

**Causes:**

- Python not installed
- Python not added to system PATH
- Using wrong Python command

**Solutions:**

**Windows:**

1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Restart command prompt
4. Test: `python --version`

**Alternative Windows solution:**

```bash
# Try these commands instead
py --version
python3 --version
```

**macOS:**

```bash
# Install using Homebrew
brew install python3

# Or use python3 command
python3 --version
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Issue: "pip not found" or "No module named pip"

**Symptoms:**

```bash
'pip' is not recognized as an internal or external command
pip: command not found
```

**Solutions:**

**Windows:**

```bash
# Try these alternatives
py -m pip --version
python -m pip --version
python3 -m pip --version
```

**Install pip manually:**

```bash
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

**macOS/Linux:**

```bash
# Install pip
sudo apt install python3-pip  # Ubuntu/Debian
brew install python3          # macOS with Homebrew
```

### Issue: "Permission denied" during installation

**Symptoms:**

```bash
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solutions:**

**Option 1: Use virtual environment (Recommended)**

```bash
python -m venv feedback_env
# Windows
feedback_env\Scripts\activate
# macOS/Linux
source feedback_env/bin/activate
pip install -r requirements.txt
```

**Option 2: User installation**

```bash
pip install --user -r requirements.txt
```

**Option 3: Administrator/sudo (Not recommended)**

```bash
# Windows (Run as Administrator)
pip install -r requirements.txt

# macOS/Linux
sudo pip install -r requirements.txt
```

### Issue: Package installation fails with dependency conflicts

**Symptoms:**

```bash
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solutions:**

**Clean installation:**

```bash
# Create fresh virtual environment
python -m venv fresh_env
# Activate it
fresh_env\Scripts\activate  # Windows
source fresh_env/bin/activate  # macOS/Linux

# Upgrade pip
pip install --upgrade pip

# Install packages one by one
pip install torch
pip install transformers==4.53.0
pip install streamlit
pip install datasets>=2.18.0
pip install scikit-learn pandas accelerate>=0.26.0 tqdm
```

**Force reinstall:**

```bash
pip install --force-reinstall -r requirements.txt
```

---

## 🎓 Training Problems

### Issue: "CUDA out of memory" during training

**Symptoms:**

```bash
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

**Reduce batch size:**

```python
# In finetune_classifier.py, change:
per_device_train_batch_size=8,  # Reduce from 16
per_device_eval_batch_size=32,  # Reduce from 64
```

**Use gradient accumulation:**

```python
# Add to training arguments:
gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
```

**Force CPU training:**

```python
# In finetune_classifier.py, add:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
```

**Clear GPU cache:**

```python
# Add to training script:
import torch
torch.cuda.empty_cache()
```

### Issue: Training is extremely slow

**Symptoms:**

- Training takes hours instead of minutes
- Each epoch takes 30+ minutes

**Causes:**

- Running on CPU instead of GPU
- Batch size too small
- Too many training epochs

**Solutions:**

**Check if GPU is being used:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
```

**Optimize for CPU training:**

```python
# Reduce epochs for testing
num_train_epochs=1,  # Instead of 3

# Increase batch size if memory allows
per_device_train_batch_size=32,  # If you have enough RAM
```

**Use smaller model:**

```python
# In finetune_classifier.py, change model:
model_name = "distilbert-base-uncased"  # Faster than bert-base-uncased
```

### Issue: Training accuracy is very low

**Symptoms:**

- Final accuracy below 70%
- Model seems to predict randomly
- Loss doesn't decrease during training

**Causes:**

- Learning rate too high or too low
- Insufficient training data
- Data quality issues
- Wrong labels

**Solutions:**

**Check data quality:**

```python
# Add to training script to inspect data
import json

with open('data/feedback_classify_train.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i < 5:  # Print first 5 examples
            data = json.loads(line)
            print(f"Text: {data['text']}")
            print(f"Label: {data['label']}")
            print("---")
```

**Adjust learning rate:**

```python
# Try different learning rates
learning_rate=5e-5,  # Higher learning rate
# or
learning_rate=1e-5,  # Lower learning rate
```

**Increase training epochs:**

```python
num_train_epochs=5,  # More training
```

**Check label distribution:**

```python
from collections import Counter
import json

labels = []
with open('data/feedback_classify_train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        labels.append(data['label'])

print("Label distribution:")
for label, count in Counter(labels).items():
    print(f"{label}: {count}")
```

### Issue: "No module named 'finetune_classifier'"

**Symptoms:**

```bash
ModuleNotFoundError: No module named 'finetune_classifier'
```

**Causes:**

- Running from wrong directory
- File doesn't exist
- Python path issues

**Solutions:**

**Check current directory:**

```bash
# Make sure you're in the project root
ls -la  # Should show finetune_classifier.py
pwd     # Should show project directory path
```

**Run from correct directory:**

```bash
cd path/to/customer-feedback-analyzer
python finetune_classifier.py
```

**Check file exists:**

```bash
# Verify file exists
ls -la finetune_classifier.py
```

---

## 🤖 Model Loading Errors

### Issue: "Model not found" in Streamlit app

**Symptoms:**

```bash
FileNotFoundError: [Errno 2] No such file or directory: 'models/feedback_classifier'
```

**Causes:**

- Model hasn't been trained yet
- Model training failed
- Wrong model path

**Solutions:**

**Train the model first:**

```bash
python finetune_classifier.py
```

**Verify model files exist:**

```bash
# Check if model directory exists
ls -la models/
ls -la models/feedback_classifier/

# Should contain files like:
# config.json
# pytorch_model.bin
# tokenizer.json
# tokenizer_config.json
```

**Check model path in code:**

```python
# In inference.py, verify path:
model_dir = "models/feedback_classifier"
# Make sure this matches your actual directory structure
```

### Issue: "Can't load tokenizer" error

**Symptoms:**

```bash
OSError: Can't load tokenizer for 'models/feedback_classifier'
```

**Causes:**

- Incomplete model files
- Corrupted model files
- Version mismatch

**Solutions:**

**Retrain the model:**

```bash
# Delete existing model and retrain
rm -rf models/feedback_classifier
python finetune_classifier.py
```

**Check transformers version:**

```bash
pip show transformers
# Should be version 4.53.0 or compatible
```

**Manual verification:**

```python
# Test model loading manually
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    tokenizer = AutoTokenizer.from_pretrained("models/feedback_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("models/feedback_classifier")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
```

### Issue: Model predictions are inconsistent

**Symptoms:**

- Same input gives different outputs
- Confidence scores vary wildly
- Results don't make sense

**Causes:**

- Model not properly trained
- Random seed issues
- Input preprocessing problems

**Solutions:**

**Set random seeds:**

```python
# Add to inference.py
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Call before model loading
set_seed(42)
```

**Check model evaluation mode:**

```python
# In inference.py, ensure model is in eval mode
model.eval()
```

**Verify input preprocessing:**

```python
# Debug tokenization
text = "Sample feedback"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
print(f"Input IDs: {inputs['input_ids']}")
print(f"Attention mask: {inputs['attention_mask']}")
```

---

## 🎨 Streamlit App Issues

### Issue: Streamlit app won't start

**Symptoms:**

```bash
streamlit: command not found
```

**Solutions:**

**Install Streamlit:**

```bash
pip install streamlit
```

**Use full path:**

```bash
python -m streamlit run app.py
```

**Check installation:**

```bash
pip show streamlit
streamlit --version
```

### Issue: "Port already in use" error

**Symptoms:**

```bash
OSError: [Errno 48] Address already in use
```

**Solutions:**

**Use different port:**

```bash
streamlit run app.py --server.port 8502
```

**Kill existing processes:**

```bash
# Windows
taskkill /f /im streamlit.exe

# macOS/Linux
pkill -f streamlit
lsof -ti:8501 | xargs kill -9  # Kill process on port 8501
```

**Find and kill specific process:**

```bash
# Find process using port 8501
netstat -tulpn | grep 8501  # Linux
lsof -i :8501              # macOS

# Kill the process ID
kill -9 <PID>
```

### Issue: Streamlit app loads but shows errors

**Symptoms:**

- App loads but shows error messages
- Components don't work properly
- Styling issues

**Solutions:**

**Clear Streamlit cache:**

```bash
streamlit cache clear
```

**Check browser console:**

1. Open browser developer tools (F12)
2. Check Console tab for JavaScript errors
3. Check Network tab for failed requests

**Update Streamlit:**

```bash
pip install --upgrade streamlit
```

**Test with minimal app:**

```python
# Create test_app.py
import streamlit as st
st.write("Hello, World!")

# Run test
streamlit run test_app.py
```

### Issue: App is slow or unresponsive

**Symptoms:**

- Long loading times
- UI freezes during analysis
- Browser becomes unresponsive

**Solutions:**

**Enable caching:**

```python
# In app.py, add caching to model loading
@st.cache_resource
def load_model():
    from inference import analyze_feedback
    return analyze_feedback

# Use cached function
analyze_func = load_model()
```

**Optimize inference:**

```python
# Reduce model precision
model = model.half()  # Use 16-bit precision

# Use smaller batch sizes
# Process text in smaller chunks
```

**Add progress indicators:**

```python
# Show progress during analysis
with st.spinner("Analyzing feedback..."):
    result = analyze_feedback(text)
```

---

## ⚡ Performance Problems

### Issue: Inference is very slow

**Symptoms:**

- Each prediction takes several seconds
- Batch processing is extremely slow

**Solutions:**

**Use GPU acceleration:**

```python
# In inference.py, ensure GPU is used if available
device = 0 if torch.cuda.is_available() else -1
```

**Optimize model loading:**

```python
# Load model once and reuse
class FeedbackAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.nlp = None
        self._load_model()

    def _load_model(self):
        if self.model is None:
            # Load model only once
            model_dir = "models/feedback_classifier"
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.nlp = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1,
            )

    def analyze(self, text):
        result = self.nlp(text)[0]
        top = max(result, key=lambda x: x["score"])
        return top["label"], top["score"]

# Use singleton pattern
analyzer = FeedbackAnalyzer()
```

**Use model quantization:**

```python
# Reduce model size and increase speed
model = model.half()  # 16-bit precision
```

**Batch processing optimization:**

```python
def analyze_batch(texts, batch_size=32):
    """Process texts in batches for better performance"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = nlp(batch)
        results.extend(batch_results)
    return results
```

### Issue: High memory usage

**Symptoms:**

- System runs out of RAM
- Other applications become slow
- Memory usage keeps increasing

**Solutions:**

**Monitor memory usage:**

```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory usage: {get_memory_usage():.1f} MB")
```

**Clear cache regularly:**

```python
import gc
import torch

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Call after processing batches
clear_memory()
```

**Use smaller model:**

```python
# Switch to DistilBERT (smaller, faster)
model_name = "distilbert-base-uncased"
```

**Limit batch sizes:**

```python
# Reduce batch size to use less memory
per_device_train_batch_size=8,
per_device_eval_batch_size=16,
```

---

## 💾 Memory Issues

### Issue: "Out of memory" errors

**Symptoms:**

```bash
RuntimeError: [enforce fail at alloc_cpu.cpp:75]
MemoryError: Unable to allocate array
```

**Solutions:**

**Reduce model size:**

```python
# Use DistilBERT instead of BERT
model_name = "distilbert-base-uncased"

# Or use model quantization
model = model.half()
```

**Process data in smaller chunks:**

```python
def process_large_dataset(texts, chunk_size=100):
    results = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        chunk_results = [analyze_feedback(text) for text in chunk]
        results.extend(chunk_results)

        # Clear memory after each chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
```

**Increase system virtual memory:**

- **Windows**: Control Panel → System → Advanced → Performance Settings → Virtual Memory
- **Linux**: Add swap space
- **macOS**: System manages automatically

**Monitor memory usage:**

```python
import psutil

def check_memory():
    memory = psutil.virtual_memory()
    print(f"Total: {memory.total / 1024**3:.1f} GB")
    print(f"Available: {memory.available / 1024**3:.1f} GB")
    print(f"Used: {memory.percent}%")

check_memory()
```

---

## 🖥️ GPU/CUDA Problems

### Issue: "CUDA not available" when GPU is installed

**Symptoms:**

```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**

**Check CUDA installation:**

```bash
nvidia-smi  # Should show GPU info
nvcc --version  # Should show CUDA version
```

**Install CUDA-compatible PyTorch:**

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA version (check CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify installation:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Issue: "CUDA version mismatch"

**Symptoms:**

```bash
RuntimeError: The NVIDIA driver on your system is too old
```

**Solutions:**

**Update NVIDIA drivers:**

1. Visit [NVIDIA Driver Downloads](https://www.nvidia.com/drivers/)
2. Download latest driver for your GPU
3. Install and restart system

**Check compatibility:**

```bash
# Check driver version
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**Install compatible PyTorch version:**

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 📊 Data Issues

### Issue: "Invalid JSON" in training data

**Symptoms:**

```bash
json.decoder.JSONDecodeError: Expecting ',' delimiter
```

**Solutions:**

**Validate JSON format:**

```python
import json

def validate_jsonl(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error on line {i}: {e}")
                print(f"Line content: {line}")

validate_jsonl('data/feedback_classify_train.jsonl')
```

**Fix common JSON issues:**

```python
def fix_jsonl_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                try:
                    # Parse and rewrite to ensure valid JSON
                    data = json.loads(line)
                    f_out.write(json.dumps(data) + '\n')
                except json.JSONDecodeError:
                    print(f"Skipping invalid line: {line}")

fix_jsonl_file('data/feedback_classify_train.jsonl', 'data/fixed_train.jsonl')
```

### Issue: Unbalanced dataset

**Symptoms:**

- Some categories have very few examples
- Model performs poorly on minority classes

**Solutions:**

**Check class distribution:**

```python
from collections import Counter
import json

def analyze_dataset(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            labels.append(data['label'])

    counter = Counter(labels)
    total = len(labels)

    print("Class distribution:")
    for label, count in counter.most_common():
        percentage = (count / total) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")

analyze_dataset('data/feedback_classify_train.jsonl')
```

**Balance the dataset:**

```python
def balance_dataset(input_file, output_file, min_samples=50):
    # Read all data
    data_by_label = {}
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            label = item['label']
            if label not in data_by_label:
                data_by_label[label] = []
            data_by_label[label].append(item)

    # Balance by duplicating minority classes
    balanced_data = []
    for label, items in data_by_label.items():
        while len(items) < min_samples:
            items.extend(items[:min_samples - len(items)])
        balanced_data.extend(items[:min_samples])

    # Shuffle and write
    import random
    random.shuffle(balanced_data)

    with open(output_file, 'w') as f:
        for item in balanced_data:
            f.write(json.dumps(item) + '\n')

balance_dataset('data/feedback_classify_train.jsonl', 'data/balanced_train.jsonl')
```

---

## 🚀 Deployment Issues

### Issue: Streamlit Cloud deployment fails

**Symptoms:**

- App fails to build on Streamlit Cloud
- Dependencies not found
- Import errors

**Solutions:**

**Check requirements.txt:**

```txt
# Make sure all dependencies are listed with versions
transformers==4.53.0
torch>=1.9.0
streamlit>=1.28.0
datasets>=2.18.0
scikit-learn>=1.0.0
pandas>=1.3.0
accelerate>=0.26.0
tqdm>=4.62.0
```

**Add packages.txt for system dependencies:**

```txt
# Create packages.txt in root directory
build-essential
```

**Check Python version:**

```toml
# Create .streamlit/config.toml
[server]
headless = true
port = $PORT
enableCORS = false

[theme]
base = "light"
```

**Test locally first:**

```bash
# Test exact deployment environment
pip install -r requirements.txt
streamlit run app.py
```

### Issue: Heroku deployment fails

**Symptoms:**

- Build fails during deployment
- App crashes on startup
- Memory limit exceeded

**Solutions:**

**Create Procfile:**

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

**Add runtime.txt:**

```
python-3.9.18
```

**Optimize for Heroku:**

```python
# In app.py, add memory optimization
import os

# Use smaller model for deployment
if os.environ.get('DYNO'):  # Running on Heroku
    model_name = "distilbert-base-uncased"
else:
    model_name = "bert-base-uncased"
```

**Check slug size:**

```bash
# Heroku has 500MB limit
# Use smaller dependencies if needed
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 🔍 Advanced Debugging

### Enable Debug Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use in your code
logger.debug("Loading model...")
logger.info("Model loaded successfully")
logger.error("Failed to load model: %s", str(e))
```

### Profile Performance

```python
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    """Profile a function's performance"""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

    return result

# Usage
result = profile_function(analyze_feedback, "Sample text")
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def analyze_with_memory_tracking(text):
    return analyze_feedback(text)

# Run with: python -m memory_profiler your_script.py
```

### Test Individual Components

```python
def test_tokenizer():
    """Test tokenizer separately"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("models/feedback_classifier")

        test_text = "This is a test"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"Tokenization successful: {tokens}")
        return True
    except Exception as e:
        print(f"Tokenizer test failed: {e}")
        return False

def test_model_loading():
    """Test model loading separately"""
    try:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained("models/feedback_classifier")
        print("Model loading successful")
        return True
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

def test_inference():
    """Test inference pipeline"""
    try:
        from inference import analyze_feedback
        result = analyze_feedback("Test feedback")
        print(f"Inference successful: {result}")
        return True
    except Exception as e:
        print(f"Inference failed: {e}")
        return False

# Run all tests
if __name__ == "__main__":
    print("Running component tests...")
    test_tokenizer()
    test_model_loading()
    test_inference()
```

### Environment Debugging

```python
def debug_environment():
    """Print environment information for debugging"""
    import sys
    import torch
    import transformers
    import streamlit
    import platform

    print("=== Environment Debug Info ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Streamlit version: {streamlit.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    print("\n=== File System ===")
    import os
    print(f"Current directory: {os.getcwd()}")
    print(f"Model directory exists: {os.path.exists('models/feedback_classifier')}")
    if os.path.exists('models/feedback_classifier'):
        print(f"Model files: {os.listdir('models/feedback_classifier')}")

debug_environment()
```

---

## 📞 Getting Help

If you've tried all the solutions above and still have issues:

### 1. Check Existing Issues

- Search [GitHub Issues](https://github.com/your-repo/issues)
- Look for similar problems and solutions

### 2. Create a Detailed Bug Report

Include:

- **Environment info** (run `debug_environment()` above)
- **Exact error message** (copy-paste full traceback)
- **Steps to reproduce** the issue
- **What you've already tried**

### 3. Community Support

- Join our Discord/Slack community
- Ask questions in discussions
- Share your solutions to help others

### 4. Professional Support

- Email: support@your-project.com
- Priority support for enterprise users
- Custom training and deployment assistance

---

**Remember**: Most issues have simple solutions. Take your time, read error messages carefully, and don't hesitate to ask for help! 🚀
