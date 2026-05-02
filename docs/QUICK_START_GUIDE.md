# Quick Start Guide for Non-Technical Users

## Welcome! 👋

This project helps you **identify smells using gas sensor data**. Think of it like an "electronic nose" that can tell what odor a sensor is smelling.

**No technical background needed!** This guide will walk you through everything step-by-step.

---

## What Will You See?

When you open the app, you'll see:

1. **A prediction tool** - Upload a sensor file and get smell predictions
2. **Visual charts** - See how sensors respond to different smells
3. **Confidence scores** - Know how sure the model is about its prediction
4. **Research insights** - Understand what the model found interesting

---

## Step-by-Step: Your First Prediction

### Step 1: Open the App
Run this command in your terminal:
```bash
uv run streamlit run src/app/streamlit_app.py
```

The app will open in your web browser automatically.

### Step 2: Choose a Sample (Recommended for First Time)
On the left sidebar:
- Look for **"Sample CSV"** dropdown
- Select any sample from the list (e.g., `allspice_6.csv`)
- The app loads real sensor data from the SmellNet dataset

**Why start with a sample?**
- It's guaranteed to work
- You'll see what good sensor data looks like
- No need to prepare your own files yet

### Step 3: Pick a Model
In the sidebar under **"Model artifact"**:
- You'll see several options like `baseline_windowed_trialmax`
- The app automatically selects the **best performing model**
- Each shows its accuracy (e.g., "trial top-1 64.0%")

**What does this mean?**
- Higher percentage = better at identifying smells
- The default choice is already the best one available

### Step 4: View Results
Click on the **"Prediction Results"** tab at the top.

You'll see:
- **Predicted smell class** - What the model thinks the smell is
- **Confidence score** - How certain the model is (0-100%)
- **Top-5 alternatives** - Other possible smells ranked by likelihood
- **Sensor curves** - Visual graphs showing sensor responses

### Step 5: Understand the Prediction

#### Confidence Levels
| Confidence | What It Means | What To Do |
|------------|---------------|------------|
| **High (70%+)** | Model is very sure | Trust the prediction |
| **Medium (40-70%)** | Model sees multiple possibilities | Check top-5 list |
| **Low (<40%)** | Signal is unclear | Treat as exploratory only |

#### Reading the Top-5 List
The bar chart shows:
- **Tallest bar** = Most likely smell
- **Other bars** = Alternative possibilities
- Even if confidence is low, the correct smell might be in top-5

---

## Common Questions

### ❓ "What am I looking at?"
Each CSV file contains readings from **multiple gas sensors** over time. When exposed to a smell, each sensor responds differently - some rise quickly, others slowly, some drift up or down.

The model learns these **response patterns** to identify smells.

### ❓ "Why multiple windows?"
Instead of looking at the whole file at once, the model:
1. Cuts the sensor recording into small time windows
2. Analyzes each window separately
3. Combines all the results for a final prediction

This is like reading a book paragraph by paragraph instead of judging it by the cover.

### ❓ "What if my upload doesn't work?"
Common issues:
- **Missing sensors**: Your file needs the same sensor columns as the training data
- **Wrong format**: Must be a CSV with time column and numeric sensor readings
- **Solution**: Try a sample file first to see the expected format

### ❓ "Can I trust the prediction?"
Remember:
- ✅ This is a **research demo**, not a medical/safety device
- ✅ High confidence suggests reliable prediction
- ✅ Low confidence means "I'm not sure" - which is honest and useful
- ❌ Do NOT use for food safety, allergen detection, or hazardous gas warnings

---

## Navigation Map

Here's where to find everything:

### Sidebar (Left Panel)
| Option | Purpose |
|--------|---------|
| **Navigation** | Switch between Guide and Prediction Demo |
| **Model artifact** | Choose which trained model to use |
| **Sample CSV** | Pick a demo file or upload your own |
| **Window aggregation** | Advanced: how to combine window results |

### Main Tabs (Top)
| Tab | When to Use It |
|-----|----------------|
| **Overview** | Start here - shows model health and input status |
| **Prediction Results** | See the smell prediction and confidence |
| **Signal Analysis** | Inspect raw sensor curves before prediction |
| **Research Evidence** | Read about model performance and limitations |
| **CSV Details** | Check your file structure and sensor columns |
| **Models** | Compare different saved models side-by-side |
| **Method** | Understand how preprocessing and features work |

---

## Visual Guide: Understanding Sensor Curves

When you see the line graphs:

### What Each Line Represents
- **Each colored line** = One gas sensor channel
- **X-axis** = Time (as the smell exposure continues)
- **Y-axis** = Sensor response (how strongly it detects something)

### What To Look For
1. **Rising curves** - Sensor detecting the smell
2. **Plateau** - Sensor reaching saturation
3. **Drift** - Gradual change over time (common in gas sensors)
4. **Different shapes** - Each smell creates unique patterns across sensors

### Example Patterns
```
Strong Response:    /¯¯¯¯¯\      Weak Response:   /¯¯\
                   /       \                    /    \
                  /         \                  /      \
Time ----------->                            Time ----->
```

---

## Trying Your Own Data

### What Kind of Files Work?
Your CSV should have:
- ✅ A time column (timestamps or sequential numbers)
- ✅ Multiple sensor columns with numeric readings
- ✅ Similar format to SmellNet dataset
- ✅ One continuous recording session

### How to Upload
1. Go to **"Upload gas sensor CSV"** area
2. Drag and drop your file or click to browse
3. Wait for schema validation (green checkmark = good!)
4. Click **"Analyze smell signal"** button

### What If Schema Validation Fails?
The app will show which sensor columns are missing. Your file needs the same sensors that the model was trained on.

---

## Interpreting Results Responsibly

### ✅ Good Uses
- Research exploration
- Learning about machine olfaction
- Comparing sensor responses across samples
- Understanding ML model behavior

### ❌ NOT Suitable For
- Food safety decisions
- Allergen detection
- Medical diagnosis
- Hazardous gas warnings
- Quality control in production

**Why?** The model is trained on lab conditions with specific sensors. Real-world environments introduce variables (humidity, temperature, sensor aging) that affect accuracy.

---

## Next Steps After Your First Prediction

1. **Try different samples** - Compare how various smells look
2. **Check Signal Analysis tab** - See raw curves before model processing
3. **Read Research Evidence** - Learn which smells are hardest to predict
4. **Compare Models** - Use the Models tab to see alternative predictions
5. **Upload your own data** - Once comfortable with the format

---

## Glossary of Terms

| Term | Simple Explanation |
|------|-------------------|
| **Trial** | One complete sensor recording session |
| **Window** | A small time segment cut from a trial |
| **Feature** | A measurable pattern extracted from sensor data |
| **Classification** | Assigning a smell label to the input |
| **Confidence** | How certain the model is (0-100%) |
| **Top-5 Accuracy** | Whether correct smell appears in top 5 guesses |
| **Sensor Drift** | Gradual change in sensor readings over time |
| **Warm-up** | Initial unstable period when sensor is first exposed |
| **Aggregation** | Combining multiple window predictions into one |
| **Schema** | The required column structure for your CSV |

---

## Getting Help

If you're stuck:
1. Try a sample file first
2. Check the **CSV Details** tab for format issues
3. Read error messages carefully - they explain what's wrong
4. Review the **Project Guide** in the app for more details

---

## Summary Checklist

Before running a prediction:
- [ ] Selected a sample or uploaded a CSV
- [ ] Confirmed schema validation (green checkmark)
- [ ] Chosen a model from sidebar
- [ ] Understand confidence levels
- [ ] Ready to interpret results responsibly

You're all set! 🎉
