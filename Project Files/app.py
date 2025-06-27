import streamlit as st
import pandas as pd
import plotly.express as px
from utils import init_granite_model, get_sample_patient_data, get_patient_profile
import datetime

# --- Streamlit Application Configuration ---
# st.set_page_config() MUST be the very first Streamlit command in your script.
# No other st. commands (like st.title(), st.sidebar.title()) should come before it.
st.set_page_config(
    page_title="HealthAI: Intelligent Healthcare Assistant",
    page_icon="⚕️",
    layout="wide"
)

# Initialize the IBM Granite model once
# This part is placed after st.set_page_config but before other UI elements,
# as it's a critical backend setup that might take time or fail.
try:
    granite_model = init_granite_model()
except ValueError as e:
    st.error(f"Configuration Error: {e}. Please ensure your .env file is correctly set up.")
    granite_model = None # Set to None so the app doesn't crash immediately

# --- Main Application Title and Sidebar ---
st.title("⚕️ HealthAI: Intelligent Healthcare Assistant")

# Sidebar for Navigation and Patient Profile
st.sidebar.title("Navigation")
feature_selection = st.sidebar.radio(
    "Choose a feature:",
    ("Patient Chat", "Disease Prediction", "Treatment Plans", "Health Analytics")
)

st.sidebar.header("Patient Profile")
patient_profile = get_patient_profile()
st.sidebar.write(f"**Age:** {patient_profile['age']}")
st.sidebar.write(f"**Gender:** {patient_profile['gender']}")
st.sidebar.write(f"**Medical History:** {patient_profile['medical_history']}")

# --- Feature Implementations ---

if feature_selection == "Patient Chat":
    st.header("Patient Chat")
    st.write("Ask any health-related question and get an empathetic, evidence-based response.")

    if granite_model:
        # Initialize chat history in session state if not present
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input for user query
        query = st.chat_input("Type your health question here...")
        if query:
            # Add user query to chat history and display it
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Get response from the model
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Prompt for Patient Chat based on document's example
                    query_prompt = f"""As a healthcare AI assistant, provide a helpful, accurate, and evidence-based response to the following patient question:

PATIENT QUESTION: {query}

Provide a clear, empathetic response that:
- Directly addresses the question
- Includes relevant medical facts
- Acknowledges limitations (when appropriate)
- Suggests when to seek professional medical advice
- Avoids making definitive diagnoses
- Uses accessible, non-technical language

RESPONSE:
"""
                    response = granite_model.generate_text(prompt=query_prompt)
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        st.warning("Granite model not initialized. Please check your configuration in the .env file and utils.py.")


elif feature_selection == "Disease Prediction":
    st.header("Disease Prediction")
    st.write("Enter your symptoms to get potential condition predictions.")

    if granite_model:
        with st.form("disease_prediction_form"):
            symptoms = st.text_area("Describe your symptoms (e.g., persistent headache, fatigue, mild fever):", height=150)
            # Pre-fill age and gender from patient profile
            age = st.number_input("Your Age:", min_value=0, max_value=120, value=patient_profile['age'])
            gender = st.selectbox("Your Gender:", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(patient_profile['gender']))
            medical_history = st.text_area("Brief Medical History (optional):", value=patient_profile['medical_history'])

            submitted = st.form_submit_button("Get Prediction")

            if submitted and symptoms:
                st.write("Analyzing your symptoms...")
                # Fetch recent health metrics (example data for the prompt)
                recent_health_data = get_sample_patient_data().tail(1) # Get the most recent day's data
                avg_heart_rate = recent_health_data['Heart Rate'].values[0] if not recent_health_data.empty else 'N/A'
                avg_systolic_bp = recent_health_data['Systolic BP'].values[0] if not recent_health_data.empty else 'N/A'
                avg_diastolic_bp = recent_health_data['Diastolic BP'].values[0] if not recent_health_data.empty else 'N/A'
                avg_blood_glucose = recent_health_data['Blood Glucose'].values[0] if not recent_health_data.empty else 'N/A'

                # Prompt for Disease Prediction based on document's example
                prediction_prompt = f"""As a medical AI assistant, predict potential health conditions based on the following patient data:

Current Symptoms: {symptoms}
Age: {age}
Gender: {gender}
Medical History: {medical_history}

Recent Health Metrics:
- Average Heart Rate: {avg_heart_rate} bpm
- Average Blood Pressure: {avg_systolic_bp}/{avg_diastolic_bp} mmHg
- Average Blood Glucose: {avg_blood_glucose} mg/dL
- Recently Reported Symptoms: {symptoms}

Format your response as:
1. Potential condition name
2. Likelihood (High/Medium/Low)
3. Brief explanation
4. Recommended next steps

Provide the top 3 most likely conditions based on the data provided.
"""
                with st.spinner("Generating prediction..."):
                    response = granite_model.generate_text(prompt=prediction_prompt)
                st.subheader("Potential Conditions:")
                st.markdown(response)
            elif submitted and not symptoms:
                st.warning("Please enter your symptoms to get a prediction.")
    else:
        st.warning("Granite model not initialized. Please check your configuration in the .env file and utils.py.")


elif feature_selection == "Treatment Plans":
    st.header("Treatment Plans")
    st.write("Receive personalized treatment recommendations for a diagnosed condition.")

    if granite_model:
        with st.form("treatment_plan_form"):
            condition = st.text_input("Diagnosed Condition:")
            # Pre-fill age and gender from patient profile
            age = st.number_input("Patient Age:", min_value=0, max_value=120, value=patient_profile['age'])
            gender = st.selectbox("Patient Gender:", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(patient_profile['gender']))
            medical_history = st.text_area("Patient Medical History:", value=patient_profile['medical_history'])

            submitted = st.form_submit_button("Generate Treatment Plan")

            if submitted and condition:
                st.write("Generating personalized treatment plan...")
                # Prompt for Treatment Plan based on document's example
                treatment_prompt = f"""
You are a reliable medical AI assistant. Based on the following patient information, generate a complete treatment plan.

Patient Condition: {condition}  
Age: {age}  
Gender: {gender}  
Medical History: {medical_history}  

Please follow this **structured format strictly**:

1. **Recommended Medications**: Include medicine names (not placeholders), standard dosage, frequency, and purpose. Example: *Paracetamol 500mg twice daily for fever*.
2. **Lifestyle Modifications**: At least 2 suggestions.
3. **Follow-up Testing and Monitoring**: Include test names and recommended intervals.
4. **Dietary Recommendations**: Specific foods to prefer or avoid.
5. **Physical Activity Guidelines**: Example: *30 minutes brisk walking daily unless contraindicated*.
6. **Mental Health Considerations**: Simple stress-reduction or psychological support ideas.

⚠️ Do NOT use fake medicine names like abc, xyz, or medicine-1. Only include real, general medical advice or clearly say: “Consult a physician for medicine specifics”.

Begin your structured plan below:
"""

                with st.spinner("Generating treatment plan..."):
                    response = granite_model.generate_text(prompt=treatment_prompt)
                st.subheader(f"Treatment Plan for {condition}:")
                st.markdown(response)
            elif submitted and not condition:
                st.warning("Please enter a diagnosed condition to generate a treatment plan.")
    else:
        st.warning("Granite model not initialized. Please check your configuration in the .env file and utils.py.")


elif feature_selection == "Health Analytics":
    st.header("Health Analytics Dashboard")
    st.write("Visualize your vital signs and receive AI-generated insights.")

    patient_data = get_sample_patient_data()

    # Display charts
    st.subheader("Health Metric Trends Over Time")
    # Heart Rate Trend Line Chart
    fig_hr = px.line(patient_data, x='Date', y='Heart Rate', title='Heart Rate Trend')
    st.plotly_chart(fig_hr, use_container_width=True)

    # Blood Pressure Dual-Line Chart
    fig_bp = px.line(patient_data, x='Date', y=['Systolic BP', 'Diastolic BP'], title='Blood Pressure Trend')
    st.plotly_chart(fig_bp, use_container_width=True)

    # Blood Glucose Trend Line Chart with Reference Line
    fig_bg = px.line(patient_data, x='Date', y='Blood Glucose', title='Blood Glucose Trend')
    # Add a reference line for normal blood glucose (e.g., <100 mg/dL fasting)
    fig_bg.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="Normal Max (Fasting)")
    st.plotly_chart(fig_bg, use_container_width=True)

    # Symptom Frequency Pie Chart (Conceptual - requires actual symptom data)
    # For a real app, you'd collect symptom data over time to create this.
    # Here's a placeholder if you had symptom counts:
    # symptom_counts = pd.DataFrame({'Symptom': ['Headache', 'Fatigue', 'Fever', 'Nausea'], 'Count': [15, 10, 5, 3]})
    # fig_symptom = px.pie(symptom_counts, values='Count', names='Symptom', title='Symptom Frequency')
    # st.plotly_chart(fig_symptom, use_container_width=True)


    st.subheader("Key Health Indicators Summary")
    # Display key metrics with trend deltas and color-coding (simplified)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        latest_hr = patient_data['Heart Rate'].iloc[-1]
        previous_hr = patient_data['Heart Rate'].iloc[-2] if len(patient_data) > 1 else latest_hr
        delta_hr = latest_hr - previous_hr
        st.metric(label="Latest Heart Rate", value=f"{latest_hr} bpm", delta=f"{delta_hr:.1f} bpm")
        st.markdown(f"**Average:** {patient_data['Heart Rate'].mean():.1f} bpm")

    with col2:
        latest_sys = patient_data['Systolic BP'].iloc[-1]
        previous_sys = patient_data['Systolic BP'].iloc[-2] if len(patient_data) > 1 else latest_sys
        delta_sys = latest_sys - previous_sys
        st.metric(label="Latest Systolic BP", value=f"{latest_sys} mmHg", delta=f"{delta_sys:.1f} mmHg")
        st.markdown(f"**Average:** {patient_data['Systolic BP'].mean():.1f} mmHg")

    with col3:
        latest_dia = patient_data['Diastolic BP'].iloc[-1]
        previous_dia = patient_data['Diastolic BP'].iloc[-2] if len(patient_data) > 1 else latest_dia
        delta_dia = latest_dia - previous_dia
        st.metric(label="Latest Diastolic BP", value=f"{latest_dia} mmHg", delta=f"{delta_dia:.1f} mmHg")
        st.markdown(f"**Average:** {patient_data['Diastolic BP'].mean():.1f} mmHg")

    with col4:
        latest_bg = patient_data['Blood Glucose'].iloc[-1]
        previous_bg = patient_data['Blood Glucose'].iloc[-2] if len(patient_data) > 1 else latest_bg
        delta_bg = latest_bg - previous_bg
        st.metric(label="Latest Blood Glucose", value=f"{latest_bg} mg/dL", delta=f"{delta_bg:.1f} mg/dL")
        st.markdown(f"**Average:** {patient_data['Blood Glucose'].mean():.1f} mg/dL")


    st.subheader("AI-Generated Insights")
    if granite_model:
        with st.spinner("Generating insights..."):
            # A simple prompt for general insights based on recent trends.
            # For a real app, this would involve more sophisticated data analysis
            # before passing to the LLM.
            analytics_prompt = f"""Based on the following recent health metrics for a patient (last 7 days):
Heart Rates: {list(patient_data['Heart Rate'].tail(7))}
Systolic BPs: {list(patient_data['Systolic BP'].tail(7))}
Diastolic BPs: {list(patient_data['Diastolic BP'].tail(7))}
Blood Glucoses: {list(patient_data['Blood Glucose'].tail(7))}

Provide a brief summary of potential health observations or general recommendations for improvement based on these trends. Avoid making definitive diagnoses.
"""
            response = granite_model.generate_text(prompt=analytics_prompt)
            st.markdown(response)
    else:
        st.warning("Granite model not initialized. Please check your configuration in the .env file and utils.py.")