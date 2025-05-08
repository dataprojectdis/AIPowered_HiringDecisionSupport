import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import subprocess
import shap
import matplotlib.pyplot as plt

# ---------------------- Utility ----------------------
def clean_prompt(text):
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")

# ---------------------- LLM Bias Explanation Function ----------------------
def generate_bias_explanation(bias_df, bias_type):
    prompt = f"""
You are an AI hiring assistant analyzing potential bias in an AI-powered hiring system. 
Your goal is to help recruiters understand how the model may be scoring different groups unequally, 
but in a friendly and non-technical way.

Here is the data showing the average predicted hire score for each group under "{bias_type}":

{bias_df.to_markdown(index=False)}

Please write a short explanation that:
- Clearly summarizes which group scored higher or lower
- Avoids technical language like "model weights" or "statistical significance"
- Sounds conversational, as if explaining this to a recruiter in a meeting
- Encourages responsible hiring without being accusatory
"""

    try:
        prompt = clean_prompt(prompt)
        process = subprocess.Popen(
            ['ollama', 'run', 'mistral', prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return f"Ollama Error: {stderr.strip()}"
        return stdout.strip()

    except Exception as e:
        return f"Error generating bias explanation: {e}"

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="AI Hiring Score Generator", layout="wide")

st.title("📄 AI Hiring Score Generator")

uploaded_file = st.file_uploader("Upload CSV File with Candidate Data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head())

    try:
        education_map = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
        df['EducationEncoded'] = df['Education'].map(education_map)

        df['HireScore'] = (
            df['GPA'] * 25 +
            df['Experience (Years)'] * 3 +
            df['EducationEncoded'] * 5 +
            df['No. of Projects'] * 2 +
            df['Certifications'] * 2
        )

        features = ['GPA', 'Experience (Years)', 'EducationEncoded', 'Certifications', 'No. of Projects']
        X = df[features]
        y = df['HireScore']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        df['PredictedHireScore'] = model.predict(X)

        st.subheader("🏆 Top Candidates by Predicted Hire Score")
        st.dataframe(df.sort_values(by='PredictedHireScore', ascending=False)[
            ['Name', 'Gender', 'GPA', 'Experience (Years)', 'Education', 'No. of Projects', 'PredictedHireScore']
        ].head(10))

        st.subheader("📈 Feature Importance")
        importances = dict(zip(X.columns, model.feature_importances_))
        importance_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

        st.subheader("🧠 Bias Analysis in Predicted Hire Scores")

        bias_option = st.selectbox(
            "Select a group to check for potential bias:",
            options=["Gender", "Education", "Experience"]
        )

        if bias_option == "Gender":
            group_col = "Gender"
        elif bias_option == "Education":
            group_col = "Education"
        elif bias_option == "Experience":
            df['ExperienceBucket'] = pd.cut(
                df['Experience (Years)'],
                bins=[0, 2, 5, float('inf')],
                labels=["0–2 years", "3–5 years", "6+ years"]
            )
            group_col = "ExperienceBucket"

        bias_df = df.groupby(group_col)['PredictedHireScore'].mean().reset_index()
        bias_df.columns = [group_col, 'Average Predicted Score']

        st.write(f"📊 Average Predicted Hire Score by {group_col}:")
        st.dataframe(bias_df)
        st.bar_chart(bias_df.set_index(group_col))

        if 'bias_explanations' not in st.session_state:
            st.session_state.bias_explanations = {}

        explanation = st.session_state.bias_explanations.get(bias_option, None)

        if explanation:
            st.subheader("💡 AI Assistant Explanation")
            st.write(explanation)
        else:
            if st.button("Generate Bias Explanation with AI"):
                with st.spinner("Analyzing bias using AI assistant..."):
                    explanation = generate_bias_explanation(bias_df, bias_option)
                    st.session_state.bias_explanations[bias_option] = explanation
                st.subheader("💡 AI Assistant Explanation")
                st.write(explanation)

        # ------------------ SHAP + LLM Candidate Dropdown ------------------
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        st.subheader("🔎 SHAP + LLM: Explain a Candidate")

        if 'candidate_explanations' not in st.session_state:
            st.session_state.candidate_explanations = {}

        candidate_names = list(df['Name'])
        selected_candidate = st.selectbox("Select a candidate:", ["-- Select --"] + candidate_names)

        if selected_candidate != "-- Select --" and selected_candidate in st.session_state.candidate_explanations:
            explanation_ready = True
        else:
            explanation_ready = False

        if selected_candidate and selected_candidate != "-- Select --":
            i = df[df['Name'] == selected_candidate].index[0]
            row = df.loc[i]

            st.markdown(f"### 👤 Candidate: {selected_candidate}")
            st.write(row[features])

            st.markdown("**Feature Impact on Hire Score:**")
            shap.plots.bar(shap_values[i], show=False)
            st.pyplot(plt.gcf())
            plt.clf()

            if not explanation_ready:
                if st.button("Generate Candidate Explanation with AI"):
                    with st.spinner("Generating explanation with AI assistant..."):
                        top_positive = sorted(
                            zip(features, shap_values[i].values), key=lambda x: -x[1]
                        )[:2]
                        top_negative = sorted(
                            zip(features, shap_values[i].values), key=lambda x: x[1]
                        )[:1]

                        prompt = f"""
You are an AI hiring assistant helping recruiters understand why this candidate received a high hire score.
Use the following details from their profile and explain the strengths and limitations in a clear, conversational way.
Avoid numbers from model calculations. Instead, refer to the actual values (like GPA 3.8, 5 certifications, etc.).

Candidate Details:
- GPA: {row['GPA']}
- Experience: {row['Experience (Years)']} years
- Education: {row['Education']}
- Certifications: {row['Certifications']}
- Projects: {row['No. of Projects']}

From SHAP analysis, strengths include: {', '.join([f"{feat}" for feat, _ in top_positive])}.
A possible area for improvement is: {', '.join([f"{feat}" for feat, _ in top_negative])}.

Be honest in your analysis. While remaining respectful and professional, feel free to point out genuine concerns if the candidate's performance appears low. Avoid false positivity.
"""
                        try:
                            prompt = clean_prompt(prompt)
                            process = subprocess.Popen(
                                ['ollama', 'run', 'mistral', prompt],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            stdout, stderr = process.communicate()

                            if process.returncode == 0:
                                st.session_state.candidate_explanations[selected_candidate] = stdout.strip()
                            else:
                                st.warning("LLM error: " + stderr)

                        except Exception as e:
                            st.error("LLM call failed: " + str(e))

            if selected_candidate in st.session_state.candidate_explanations:
                st.markdown("**💬 AI Explanation:**")
                st.write(st.session_state.candidate_explanations[selected_candidate])

    except KeyError as e:
        st.error(f"Missing expected column: {e}")

    # ------------------ Candidate Comparison ------------------
    st.subheader("⚖️ Compare Two Candidates")

    col1, col2 = st.columns(2)
    with col1:
        candidate1 = st.selectbox("Select Candidate 1:", ["-- Select --"] + candidate_names, key="cand1")
    with col2:
        candidate2 = st.selectbox("Select Candidate 2:", ["-- Select --"] + candidate_names, key="cand2")

    if candidate1 != "-- Select --" and candidate2 != "-- Select --" and candidate1 != candidate2:
        row1 = df[df['Name'] == candidate1].iloc[0]
        row2 = df[df['Name'] == candidate2].iloc[0]

        compare_prompt = f"""
You are an AI hiring assistant. Two candidates have been shortlisted, and we need your help comparing them and selecting who would be a better fit.
Provide a professional and clear comparison using the given information. Highlight their strengths and concerns where relevant.
Conclude with a recommendation.

Candidate 1:
- Name: {candidate1}
- GPA: {row1['GPA']}
- Experience: {row1['Experience (Years)']} years
- Education: {row1['Education']}
- Certifications: {row1['Certifications']}
- Projects: {row1['No. of Projects']}
- Predicted Hire Score: {row1['PredictedHireScore']:.2f}

Candidate 2:
- Name: {candidate2}
- GPA: {row2['GPA']}
- Experience: {row2['Experience (Years)']} years
- Education: {row2['Education']}
- Certifications: {row2['Certifications']}
- Projects: {row2['No. of Projects']}
- Predicted Hire Score: {row2['PredictedHireScore']:.2f}

Please provide a reasoned recommendation between the two candidates.
"""
        if st.button("Compare Candidates with AI"):
            with st.spinner("Analyzing both candidates..."):
                try:
                    compare_prompt = clean_prompt(compare_prompt)
                    process = subprocess.Popen(
                        ['ollama', 'run', 'mistral', compare_prompt],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, stderr = process.communicate()

                    if process.returncode == 0:
                        st.markdown("**🧠 AI Comparison Result:**")
                        st.write(stdout.strip())
                    else:
                        st.warning("LLM error: " + stderr)
                except Exception as e:
                    st.error("LLM call failed: " + str(e))
