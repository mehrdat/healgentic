# 🏥 Medical Diagnosis AI System

A smart AI doctor that asks questions and helps find what might be wrong with your health.

## 📖 What This Does

This app works like talking to a doctor:
1. You tell it your symptoms
2. It asks you questions to learn more
3. It gives you a diagnosis
4. It suggests what to do next

<img src="medical_diagnosis_ai/img/first_page.png" width="400">

## 🚀 How to Start

### Step 1: Install Python
Make sure you have Python 3.8 or newer on your computer.

### Step 2: Download the Code
```bash
git clone <your-repo-url>
cd bio
```

### Step 3: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 4: Set Up AI Key
1. Get a Google API key from [Google AI Studio](https://aistudio.google.com/)
2. Create a file called `.env` in the main folder
3. Add this line to the file:
```
GOOGLE_API_KEY=your_key_here
```

### Step 5: Start the App
```bash
streamlit run medical_diagnosis_ai/src/app_interactive.py
```

## 🎯 How to Use

### 1. Fill Your Information
Enter your age, gender, and medical history.

<img src="medical_diagnosis_ai/img/prompts.png" width="400">

### 2. Describe Your Problem
Type what's wrong or what symptoms you have.

<img src="medical_diagnosis_ai/img/prompts_been_asked.png" width="400">

### 3. Answer Questions
The AI will ask you questions to understand better.

### 4. Get Your Results
See the diagnosis and treatment suggestions.

<img src="medical_diagnosis_ai/img/diagnosis.png" width="400">

<img src="medical_diagnosis_ai/img/treatment_plan.png" width="400">

## 📁 What's Inside

```
bio/
├── medical_diagnosis_ai/
│   ├── src/              # Main code
│   ├── img/              # App pictures
│   └── knowledge/        # Medical books
├── requirements.txt      # What to install
└── README.md            # This file
```

## ⚠️ Important Warning

**This is NOT a real doctor!**

- Only for learning and fun
- Always see a real doctor for health problems
- Don't use this for emergencies
- The AI can make mistakes

## 🛠️ Problems?

### App Won't Start
- Check if Python is installed
- Make sure you installed requirements
- Check your Google API key

### Questions Not Working
- Try refreshing the page
- Check your internet connection
- Make sure the AI service is working

### Wrong Diagnosis
- This is normal - AI isn't perfect
- Always check with a real doctor
- Use this as a starting point only

## 📧 Need Help?

If you have problems, ask for help or report bugs.

---

**Remember: Always see a real doctor for health problems!**