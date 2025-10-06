# 🚀 Setup Guide - Customer Feedback Analyzer

This guide will walk you through setting up the Customer Feedback Analyzer on your local machine, step by step.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux Ubuntu 18.04+
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 2GB free space
- **Internet**: Required for downloading models and dependencies

### Check Your Python Version

```bash
python --version
# Should show Python 3.8.x or higher
```

If you don't have Python installed:

- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: Use Homebrew: `brew install python3`
- **Linux**: `sudo apt-get install python3 python3-pip`

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/customer-feedback-analyzer.git
cd customer-feedback-analyzer
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv feedback_env

# Activate virtual environment
# On Windows:
feedback_env\Scripts\activate

# On macOS/Linux:
source feedback_env/bin/activate
```

**Why use virtual environments?**
Virtual environments keep your project dependencies separate from other Python projects, preventing conflicts.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- `transformers==4.53.0` - Hugging Face transformers library
- `torch` - PyTorch deep learning framework
- `streamlit` - Web app framework
- `datasets>=2.18.0` - Dataset handling
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation
- `accelerate>=0.26.0` - Training acceleration
- `tqdm` - Progress bars

### Step 4: Verify Installation

```bash
python -c "import transformers, torch, streamlit; print('All packages installed successfully!')"
```

## 🎓 Training Your First Model

### Step 1: Understand the Data

The training data is in JSONL format (JSON Lines):

```json
{"text": "The app crashes when uploading files", "label": "bug"}
{"text": "Please add dark mode", "label": "feature_request"}
```

### Step 2: Train the Model

```bash
python finetune_classifier.py
```

**What happens during training:**

1. Loads pre-trained BERT model
2. Processes your training data
3. Fine-tunes the model for feedback classification
4. Saves the trained model to `models/feedback_classifier/`

**Expected output:**

```
Loading dataset...
✓ Loaded 800 training examples
✓ Loaded 200 test examples

Initializing model...
✓ Loaded bert-base-uncased

Training...
Epoch 1/3: 100%|██████████| 50/50 [02:30<00:00]
Epoch 2/3: 100%|██████████| 50/50 [02:28<00:00]
Epoch 3/3: 100%|██████████| 50/50 [02:32<00:00]

✓ Training completed!
✓ Model saved to models/feedback_classifier/
✓ Final accuracy: 99.1%
```

**Training time estimates:**

- **CPU**: 5-10 minutes
- **GPU**: 2-3 minutes

### Step 3: Test the Model

```bash
python test.py
```

This will evaluate the model on test data and show performance metrics.

## 🎨 Launch the Web Application

### Step 1: Start the Streamlit App

```bash
streamlit run app.py
```

### Step 2: Open Your Browser

The app will automatically open at `http://localhost:8501`

If it doesn't open automatically:

1. Open your web browser
2. Navigate to `http://localhost:8501`

### Step 3: Try It Out!

1. Enter some customer feedback in the text area
2. Click "Analyze Feedback"
3. See the AI classification results!

**Example feedback to try:**

- "The app keeps crashing when I upload large files" (Should classify as Bug Report)
- "Please add a dark mode option" (Should classify as Feature Request)
- "Great customer support, very helpful!" (Should classify as Praise)

## 🔧 Troubleshooting Common Issues

### Issue 1: "Python not found"

**Problem**: Python is not installed or not in PATH
**Solution**:

- Install Python from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation
- Restart your terminal/command prompt

### Issue 2: "pip not found"

**Problem**: pip is not installed
**Solution**:

```bash
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

### Issue 3: "Permission denied" errors

**Problem**: Insufficient permissions
**Solution**:

- **Windows**: Run command prompt as Administrator
- **macOS/Linux**: Use `sudo` prefix: `sudo pip install -r requirements.txt`

### Issue 4: "CUDA out of memory"

**Problem**: GPU memory insufficient
**Solution**: Force CPU usage by editing `inference.py`:

```python
# Change this line:
device=0 if torch.cuda.is_available() else -1

# To this:
device=-1  # Always use CPU
```

### Issue 5: "Model not found" in Streamlit app

**Problem**: Model hasn't been trained yet
**Solution**:

1. Make sure you ran `python finetune_classifier.py` first
2. Check that `models/feedback_classifier/` directory exists
3. Verify it contains model files (config.json, pytorch_model.bin, etc.)

### Issue 6: Slow training on CPU

**Problem**: Training takes too long
**Solutions**:

- Reduce epochs: Edit `finetune_classifier.py` and change `num_train_epochs=1`
- Use smaller dataset: Reduce training examples
- Consider cloud training with GPU

### Issue 7: Streamlit app won't start

**Problem**: Port already in use
**Solution**:

```bash
# Try different port
streamlit run app.py --server.port 8502

# Or kill existing processes
# Windows:
taskkill /f /im streamlit.exe

# macOS/Linux:
pkill -f streamlit
```

## 🚀 Advanced Setup Options

### GPU Acceleration (Optional)

If you have an NVIDIA GPU, you can speed up training significantly:

1. **Install CUDA**: Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. **Install PyTorch with CUDA**:

```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify GPU is available**:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Docker Setup (Advanced)

For containerized deployment:

1. **Create Dockerfile**:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

2. **Build and run**:

```bash
docker build -t feedback-analyzer .
docker run -p 8501:8501 feedback-analyzer
```

### Cloud Deployment

Deploy to cloud platforms:

#### Streamlit Cloud (Free)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy with one click!

#### Heroku

1. Create `Procfile`:

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

2. Deploy:

```bash
heroku create your-app-name
git push heroku main
```

## 📊 Performance Optimization

### Memory Optimization

If you're running low on memory:

1. **Reduce batch size** in `finetune_classifier.py`:

```python
per_device_train_batch_size=8  # Reduce from 16
```

2. **Use model quantization**:

```python
model = model.half()  # Use 16-bit precision
```

3. **Clear cache regularly**:

```python
import torch
torch.cuda.empty_cache()  # If using GPU
```

### Speed Optimization

To make inference faster:

1. **Use DistilBERT** (smaller, faster model):

```python
model_name = "distilbert-base-uncased"
```

2. **Enable caching** in Streamlit:

```python
@st.cache_resource
def load_model():
    # Model loading code here
```

## 🔍 Verification Checklist

After setup, verify everything works:

- [ ] Python 3.8+ installed and accessible
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] Training completes successfully
- [ ] Model files created in `models/feedback_classifier/`
- [ ] Test script runs and shows good accuracy
- [ ] Streamlit app starts without errors
- [ ] Web interface loads at `http://localhost:8501`
- [ ] Sample feedback classifications work correctly

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for error messages in the terminal
2. **Search existing issues**: Check the GitHub issues page
3. **Create a new issue**: Include error messages and system info
4. **Join our community**: Discord/Slack for real-time help

## 🎉 Next Steps

Once setup is complete:

1. **Explore the code**: Understand how each component works
2. **Try different inputs**: Test various types of feedback
3. **Modify the categories**: Add your own classification labels
4. **Experiment with parameters**: Try different training settings
5. **Deploy to the cloud**: Share your app with others

Congratulations! You now have a fully functional AI-powered feedback analyzer running on your machine! 🚀
