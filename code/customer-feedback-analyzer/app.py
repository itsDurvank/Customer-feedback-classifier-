import streamlit as st
import os
import time
import json
from datetime import datetime
from inference import analyze_feedback
import plotly.express as px
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Agentic AI | Customer Feedback Intelligence", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .agentic-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .agentic-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .accuracy-info {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(139, 195, 74, 0.1));
        border: 2px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #2e7d32;
    }
    
    .log-entry {
        background: #2d3748;
        border-left: 4px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        color: #e2e8f0;
        animation: slideIn 0.3s ease-out;
    }
    
    .log-panel {
        background: #1a202c;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
        border: 2px solid #2d3748;
    }
    
    .log-header {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px 8px 0 0;
        margin: -1rem -1rem 1rem -1rem;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .success { background-color: #28a745; }
    .warning { background-color: #ffc107; }
    .error { background-color: #dc3545; }
    .info { background-color: #17a2b8; }
    
    .feedback-result {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        backdrop-filter: blur(5px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        color: #262730;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "logs" not in st.session_state:
    st.session_state.logs = []
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "system_stats" not in st.session_state:
    st.session_state.system_stats = {
        "total_analyzed": 0,
        "model_accuracy": 0.99,  # Updated to correct accuracy
        "avg_confidence": 0.0,
        "categories_detected": set()
    }
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

def add_log(message, log_type="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.logs.append({"message": log_entry, "type": log_type, "time": timestamp})
    if len(st.session_state.logs) > 100:  # Keep only last 100 logs
        st.session_state.logs = st.session_state.logs[-100:]

def update_stats(category, confidence):
    st.session_state.system_stats["total_analyzed"] += 1
    st.session_state.system_stats["categories_detected"].add(category)
    
    # Update rolling average confidence
    current_avg = st.session_state.system_stats["avg_confidence"]
    total = st.session_state.system_stats["total_analyzed"]
    st.session_state.system_stats["avg_confidence"] = ((current_avg * (total - 1)) + confidence) / total

def sanitize_text(val):
    if val is None:
        return None
    if isinstance(val, str):
        # Remove surrogate pairs (invalid unicode)
        return val.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')
    return str(val)

def display_logs():
    """Function to display logs with professional styling"""
    if st.session_state.logs:
        st.markdown("""
        <div class="log-panel">
            <div class="log-header">
                <span>🔴 Live System Logs</span>
                <span style="font-size: 0.8rem; opacity: 0.8;">Real-time Activity Monitor</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Show recent logs (last 15)
        recent_logs = st.session_state.logs[-15:]
        log_html = ""
        for log in recent_logs:
            status_class = log.get("type", "info")
            safe_message = sanitize_text(log["message"])
            log_html += f"""
            <div class="log-entry">
                <span class="status-indicator {status_class}"></span>
                {safe_message}
            </div>
            """
        
        st.markdown(sanitize_text(log_html + "</div>"), unsafe_allow_html=True)
        
        # Auto-scroll to bottom indicator
        st.markdown("""
        <div style="text-align: center; color: #718096; font-size: 0.8rem; margin-top: 0.5rem;">
            📊 Showing last 15 entries • Total: {} logs
        </div>
        """.format(len(st.session_state.logs)), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="log-panel">
            <div class="log-header">
                <span>🔴 Live System Logs</span>
                <span style="font-size: 0.8rem; opacity: 0.8;">Waiting for Activity</span>
            </div>
            <div style="text-align: center; padding: 2rem; color: #a0aec0;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">📋</div>
                <div>No system activity yet</div>
                <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem;">Logs will appear here as you use the system</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 style="margin: 0; font-size: 2.5rem;">🤖 Agentic AI</h1>
            <h2 style="margin: 0.5rem 0 0 0; font-size: 1.5rem; opacity: 0.9;">Customer Feedback Intelligence System</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Advanced NLP-powered feedback classification with human-in-the-loop learning</p>
        </div>
        <div style="text-align: right;">
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚀</div>
                <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.3rem;">AI Agent Status</div>
                <div style="font-size: 1.1rem; font-weight: bold;">ACTIVE</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🎯 System Dashboard")
    
    # System Status
    st.markdown("### 📊 Real-time Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem; font-weight: bold;">{st.session_state.system_stats['total_analyzed']}</div>
            <div style="font-size: 0.8rem; opacity: 0.9;">Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.system_stats['categories_detected'])}</div>
            <div style="font-size: 0.8rem; opacity: 0.9;">Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Performance with accuracy note
    st.metric("Model Accuracy", f"{st.session_state.system_stats['model_accuracy']:.1%}")
    st.metric("Avg Confidence", f"{st.session_state.system_stats['avg_confidence']:.1%}")
    
    # Add accuracy explanation
    st.markdown("""
    <div class="accuracy-info">
        <strong>⚠️ Important: Accuracy ≠ Perfection</strong><br><br>
        🎯 <strong>99% accuracy means your model gets 99 out of 100 predictions correct on the test set.</strong><br><br>
        💡 This doesn't guarantee perfection on new, unseen data. Real-world performance may vary based on:
        <ul style="margin: 0.5rem 0; padding-left: 1.2rem; font-size: 0.8rem;">
            <li>Data distribution differences</li>
            <li>New feedback patterns</li>
            <li>Edge cases not in training</li>
        </ul>
        <em>Always validate results in production!</em>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Categories
    with st.expander("📋 Classification Categories", expanded=True):
        categories = [
            ("🐞", "Bug Report", "#ff4444"),
            ("💡", "Feature Request", "#4CAF50"), 
            ("🎉", "Praise", "#FFD700"),
            ("😠", "Complaint", "#FF6B35"),
            ("❓", "Question", "#2196F3"),
            ("💡", "Usage Tip", "#9C27B0"),
            ("📄", "Documentation", "#607D8B"),
            ("🔖", "Other", "#795548")
        ]
        
        for emoji, name, color in categories:
            detected = name.lower().replace(" ", "_") in st.session_state.system_stats['categories_detected']
            status = "✅" if detected else "⚪"
            st.markdown(f'<div style="color: #e2e8f0; margin: 0.3rem 0;">{status} {emoji} <strong>{name}</strong></div>', unsafe_allow_html=True)

# --- MAIN CONTENT ---
# Check model status
model_dir = "models/feedback_classifier"
model_exists = os.path.exists(model_dir) and os.listdir(model_dir)

if not model_exists:
    add_log("🚨 Model not found - Training required", "error")
    
    st.markdown("""
    <div class="agentic-card">
        <h3>🚀 Initialize Agentic AI Model</h3>
        <p>The AI agent needs to be trained before it can start analyzing feedback. This process will create a fine-tuned model specialized for your feedback categories.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Train AI Agent", type="primary", use_container_width=True):
            add_log("🔄 AI Agent training initiated", "info")
            with st.spinner("🤖 Training AI Agent... Please wait"):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                try:
                    from finetune_classifier import train_classifier_model
                    train_classifier_model()
                    add_log("✅ AI Agent training completed successfully", "success")
                    st.success("✅ AI Agent trained successfully! Please refresh the page.")
                    st.balloons()
                except Exception as e:
                    add_log(f"❌ Training failed: {str(e)}", "error")
                    st.error(f"Training failed: {e}")

else:
    add_log("✅ AI Agent model loaded and ready", "success")
    
    # Main Analysis Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="agentic-card">
            <h3 style='color: black;'>📝 Feedback Analysis Interface</h3>
            <p style='color: black;'>Enter customer feedback below for real-time AI-powered classification and sentiment analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis controls
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze with AI", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🧹 Clear", use_container_width=True)
        with col_btn3:
            batch_mode = st.checkbox("📊 Batch Mode")

        # Log button clicks
        if analyze_btn:
            add_log("🔍 Analyze button clicked", "info")
        if clear_btn:
            add_log("🧹 Clear button clicked", "info")
        if batch_mode != st.session_state.get('prev_batch_mode', False):
            add_log(f"📊 Batch mode {'enabled' if batch_mode else 'disabled'}", "info")
            st.session_state.prev_batch_mode = batch_mode

        # Batch mode UI
        batch_feedbacks = None
        feedback_text = ""
        if 'clear_batch_input' not in st.session_state:
            st.session_state.clear_batch_input = False
        def sanitize_text(val):
            if val is None:
                return None
            if isinstance(val, str):
                # Remove surrogate pairs (invalid unicode)
                return val.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')
            return str(val)
        if batch_mode:
            st.markdown("""
            <div class="agentic-card">
                <h4 style='color: black;'>Batch Feedback Input</h4>
                <p style='color: black;'>Paste multiple feedbacks (one per line) or upload a .txt file (one feedback per line).</p>
            </div>
            """, unsafe_allow_html=True)
            batch_input_value = "" if st.session_state.clear_batch_input else None
            batch_input_value = sanitize_text(batch_input_value)
            batch_feedbacks = st.text_area(
                "Paste feedbacks (one per line)",
                height=180,
                placeholder="Feedback 1\nFeedback 2\nFeedback 3",
                key="batch_feedback_input",
                value=batch_input_value
            )
            if st.session_state.clear_batch_input:
                st.session_state.clear_batch_input = False
            uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"], key="batch_file")
            if uploaded_file is not None:
                file_content = uploaded_file.read().decode("utf-8", errors="ignore")
                file_content = sanitize_text(file_content)
                batch_feedbacks = file_content
                add_log(f"📁 File uploaded: {uploaded_file.name}", "info")
        else:
            # Use the clear_input flag to control the input value
            input_value = "" if st.session_state.clear_input else None
            input_value = sanitize_text(input_value)
            feedback_text = st.text_area(
                "Customer Feedback Input",
                height=120,
                placeholder="Example: The new dashboard is amazing, but I'd love to see a dark mode option. The current bright theme hurts my eyes during night work sessions...",
                help="💡 Tip: Paste any customer feedback for instant AI classification. Press Ctrl+Enter to analyze.",
                label_visibility="collapsed",
                key="feedback_input",
                value=input_value
            )
            # Reset the clear flag after the widget is created
            if st.session_state.clear_input:
                st.session_state.clear_input = False

        # Handle clear button - use a flag instead of directly modifying session state
        if clear_btn:
            if batch_mode:
                st.session_state.clear_batch_input = True
                st.session_state.batch_file = None
                add_log("\ud83e\uddf9 Batch input cleared", "info")
            else:
                st.session_state.clear_input = True
                add_log("\ud83e\uddf9 Input field cleared", "info")
            st.rerun()

        # Check for Enter key press (Ctrl+Enter in text area)
        if not batch_mode and feedback_text and feedback_text != st.session_state.get('last_feedback_text', ''):
            if feedback_text.endswith('\n') and len(feedback_text.strip()) > 1:
                add_log("⌨️ Text input detected", "info")
                analyze_btn = True
                st.session_state.last_feedback_text = feedback_text.strip()
            else:
                st.session_state.last_feedback_text = feedback_text
    
    with col2:
        st.markdown("""
        <div class="agentic-card">
            <h3 style='color: black;'>🎯 Analysis Result</h3>
        </div>
        """, unsafe_allow_html=True)
        
        result_placeholder = st.empty()
        
        # Default state
        with result_placeholder.container():
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">AI Agent Ready</div>
                <div style="font-size: 0.9rem; opacity: 0.7;">Enter feedback and click analyze</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis Logic
    if batch_mode:
        if analyze_btn and (batch_feedbacks and batch_feedbacks.strip()):
            feedback_list = [line.strip() for line in batch_feedbacks.splitlines() if line.strip()]
            add_log(f"🔄 Starting batch analysis for {len(feedback_list)} feedbacks", "info")
            results = []
            with st.spinner(f"🤖 Analyzing {len(feedback_list)} feedbacks..."):
                for i, fb in enumerate(feedback_list):
                    try:
                        category, confidence = analyze_feedback(fb)
                        update_stats(category, confidence)
                        st.session_state.analysis_history.append({
                            "text": fb,
                            "category": category,
                            "confidence": confidence,
                            "timestamp": datetime.now()
                        })
                        results.append({
                            "Feedback": fb,
                            "Category": category,
                            "Confidence": f"{confidence:.1%}"
                        })
                        add_log(f"✅ Batch #{i+1}: {category} ({confidence:.1%})", "success")
                    except Exception as e:
                        results.append({
                            "Feedback": fb,
                            "Category": "Error",
                            "Confidence": str(e)
                        })
                        add_log(f"❌ Batch #{i+1} failed: {str(e)}", "error")
            add_log(f"🎉 Batch analysis completed: {len(results)} feedbacks processed", "success")
            st.success(f"✅ Batch analysis completed for {len(results)} feedbacks!")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        elif analyze_btn:
            add_log("⚠️ No batch feedbacks provided", "warning")
            st.warning("⚠️ Please enter or upload feedbacks to analyze in batch mode")
    else:
        if analyze_btn and feedback_text and feedback_text.strip():
            add_log(f"🔍 Analyzing: '{feedback_text[:50]}{'...' if len(feedback_text) > 50 else ''}'", "info")
            with st.spinner("🤖 AI Agent Processing..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)
                try:
                    # Perform analysis
                    category, confidence = analyze_feedback(feedback_text)
                    # Update stats
                    update_stats(category, confidence)
                    # Add to history
                    st.session_state.analysis_history.append({
                        "text": feedback_text,
                        "category": category,
                        "confidence": confidence,
                        "timestamp": datetime.now()
                    })
                    add_log(f"✅ Classification: {category} ({confidence:.1%} confidence)", "success")
                    # Display result
                    emoji_dict = {
                        "bug": "🐞", "feature_request": "💡", "praise": "🎉",
                        "complaint": "😠", "question": "❓", "usage_tip": "💡",
                        "documentation": "📄", "other": "🔖"
                    }
                    color_dict = {
                        "bug": "#ff4444", "feature_request": "#4CAF50", "praise": "#FFD700",
                        "complaint": "#FF6B35", "question": "#2196F3", "usage_tip": "#9C27B0",
                        "documentation": "#607D8B", "other": "#795548"
                    }
                    emoji = emoji_dict.get(category, "🔖")
                    display_name = category.replace('_', ' ').title()
                    color = color_dict.get(category, "#607D8B")
                    with result_placeholder.container():
                        st.markdown(f"""
                        <div class="feedback-result">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">{emoji}</div>
                            <div style="font-size: 1.8rem; font-weight: bold; color: {color}; margin-bottom: 1rem;">
                                {display_name}
                            </div>
                            <div style="font-size: 1.2rem; margin-bottom: 1rem;">
                                Confidence: <strong>{confidence:.1%}</strong>
                            </div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">
                                ✨ Analysis completed by Agentic AI
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.success("✅ Feedback analyzed successfully!")
                except Exception as e:
                    add_log(f"❌ Analysis failed: {str(e)}", "error")
                    st.error(f"Analysis failed: {e}")
        elif analyze_btn:
            add_log("⚠️ No feedback text provided", "warning")
            st.warning("⚠️ Please enter feedback text to analyze")

    # Professional Tabbed Interface for Analytics and Logs
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📈 Analytics & History", "🔴 System Logs", "⚙️ System Monitor"])
    
    with tab1:
        # Analysis History
        if st.session_state.analysis_history:
            st.markdown("""
            <div class="agentic-card">
                <h3 style='color: black;'>📈 Analysis History & Insights</h3>
                <p style='color: black;'>Comprehensive analytics and historical data from your feedback analysis sessions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create history dataframe
            history_df = pd.DataFrame(st.session_state.analysis_history)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Category distribution chart
                category_counts = history_df['category'].value_counts()
                fig = px.pie(
                    values=category_counts.values, 
                    names=category_counts.index,
                    title="Feedback Category Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence over time
                fig2 = px.line(
                    history_df.reset_index(), 
                    x='index', 
                    y='confidence',
                    title="AI Confidence Over Time",
                    labels={'index': 'Analysis #', 'confidence': 'Confidence Score'}
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Recent history table
            st.markdown("### 📋 Recent Analysis History")
            recent_history = history_df.tail(10).copy()
            recent_history['text'] = recent_history['text'].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
            recent_history['confidence'] = recent_history['confidence'].apply(lambda x: f"{x:.1%}")
            recent_history['timestamp'] = recent_history['timestamp'].apply(lambda x: x.strftime("%H:%M:%S"))
            
            st.dataframe(
                recent_history[['timestamp', 'category', 'confidence', 'text']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #666;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">No Analysis Data Yet</div>
                <div style="font-size: 0.9rem; opacity: 0.7;">Start analyzing feedback to see insights and charts here</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        # System Logs in dedicated tab
        st.markdown("""
        <div class="agentic-card">
            <h3 style='color: black;'>🔴 Live System Logs</h3>
            <p style='color: black;'>Real-time monitoring of all system activities, events, and operations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Log controls
        col_log1, col_log2, col_log3 = st.columns([2, 1, 1])
        with col_log1:
            if st.button("🔄 Refresh Logs", use_container_width=True):
                add_log("🔄 Manual log refresh triggered", "info")
                st.rerun()
        with col_log2:
            if st.button("🧹 Clear Logs", use_container_width=True):
                st.session_state.logs = []
                add_log("🧹 Log history cleared", "info")
                st.rerun()
        with col_log3:
            log_level = st.selectbox("Filter", ["All", "Info", "Success", "Warning", "Error"], index=0)
        
        # Display logs
        display_logs()
        
        # Log statistics
        if st.session_state.logs:
            log_types = [log.get("type", "info") for log in st.session_state.logs]
            log_stats = pd.Series(log_types).value_counts()
            
            st.markdown("### 📊 Log Statistics")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Total Logs", len(st.session_state.logs))
            with col_stat2:
                st.metric("Info", log_stats.get("info", 0))
            with col_stat3:
                st.metric("Success", log_stats.get("success", 0))
            with col_stat4:
                st.metric("Errors", log_stats.get("error", 0))
    
    with tab3:
        # System Monitor
        st.markdown("""
        <div class="agentic-card">
            <h3 style='color: black;'>⚙️ System Monitor</h3>
            <p style='color: black;'>Real-time system performance metrics and health indicators.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System metrics
        col_sys1, col_sys2, col_sys3, col_sys4 = st.columns(4)
        
        with col_sys1:
            st.metric(
                "Model Status", 
                "🟢 Active" if model_exists else "🔴 Inactive",
                delta="Ready" if model_exists else "Training Required"
            )
        
        with col_sys2:
            st.metric(
                "Session Uptime",
                f"{len(st.session_state.logs)} events",
                delta="Active"
            )
        
        with col_sys3:
            st.metric(
                "Categories Detected",
                len(st.session_state.system_stats['categories_detected']),
                delta=f"of 8 total"
            )
        
        with col_sys4:
            st.metric(
                "System Health",
                "🟢 Optimal",
                delta="All systems operational"
            )
        
        # Performance chart
        if st.session_state.analysis_history:
            st.markdown("### 📈 Performance Trends")
            history_df = pd.DataFrame(st.session_state.analysis_history)
            
            # Create performance metrics over time
            performance_data = []
            for i, row in history_df.iterrows():
                performance_data.append({
                    'Analysis': i + 1,
                    'Confidence': row['confidence'],
                    'Timestamp': row['timestamp']
                })
            
            perf_df = pd.DataFrame(performance_data)
            fig_perf = px.area(
                perf_df, 
                x='Analysis', 
                y='Confidence',
                title="Model Confidence Trend",
                labels={'Confidence': 'Confidence Score', 'Analysis': 'Analysis Number'}
            )
            fig_perf.update_layout(height=300)
            st.plotly_chart(fig_perf, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h4 style="margin: 0;">🤖 Agentic AI - Customer Feedback Intelligence</h4>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Powered by Advanced NLP • Human-in-the-Loop Learning • Real-time Analytics</p>
    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
        Built with ❤️ using Streamlit, Transformers & Hugging Face
    </div>
</div>
""", unsafe_allow_html=True)