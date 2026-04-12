# Student Dropout Risk Predictor

A machine learning web app that predicts the likelihood of a student 
withdrawing from their course, based on demographic and academic factors.

🔗 **[Try the live app here](https://student-dropout-predictor-nano9rdappneg44cr5cuzdr.streamlit.app/)**

---

## Why I built this

1 in 3 students in this dataset withdrew from their course. Having worked 
directly with students in both FE and HE, I constantly see this problem up close. This project uses data science to 
understand who is most at risk and why.

---

## What the data shows

Using the Open University Learning Analytics Dataset (OULAD) — 32,593 
students across 22 courses — the analysis found:

- 31.2% of students withdrew overall
- Students from the most deprived areas withdrew at nearly 38% vs 
  26% for the least deprived — a 12 percentage point gap
- Students with disabilities withdrew at 40% vs 30%
- The strongest predictors of dropout were **region**, **deprivation 
  level**, and **credits studied** — not gender or age

The system isn't failing students because of who they are. It's failing 
them because of where they come from and what they're carrying.

---

## How it works

The app uses a **Random Forest classifier** trained on 8 features:

| Feature | What it measures |
|---|---|
| Region | Where the student is based in the UK |
| IMD band | Deprivation level of the student's area |
| Studied credits | Course load |
| Previous attempts | Number of times attempted the module |
| Highest education | Prior qualification level |
| Gender | Student gender |
| Age band | Student age group |
| Disability | Whether the student has a declared disability |

**Model accuracy: 70.1%** on held-out test data (6,519 students)

---

## The bigger picture

The same techniques used here — classification, feature engineering, 
behavioural data analysis — apply directly to:
- Customer churn prediction
- Employee retention analytics
- People analytics in HR


---

## Built with

- Python, Pandas, Matplotlib, Seaborn
- Scikit-learn (Random Forest Classifier)
- Streamlit
- OULAD Dataset (Open University)

---

*Built by Ebitimi Imomotebegha and you can find me here; | [LinkedIn](https://www.linkedin.com/in/ebitimi-imomotebegha-5a06b019a/)*
