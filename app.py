import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------
# Training data
# ---------------------------
X = np.array([
    [20, 1, 170, 60, 30],
    [25, 0, 165, 55, 45],
    [30, 1, 180, 80, 60],
    [22, 0, 158, 50, 20],
    [28, 1, 175, 70, 40],
    [35, 0, 160, 65, 50]
])

y = np.array([200, 300, 500, 150, 350, 400])

# Train model
model = LinearRegression()
model.fit(X, y)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🔥 Calories Burn Predictor (Graph + Pie Chart) 📊")

age = st.number_input("Age", 1, 100, 25)
gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
height = st.number_input("Height (cm)", 100, 220, 170)
weight = st.number_input("Weight (kg)", 30, 150, 60)
duration = st.number_input("Exercise Duration (minutes)", 1, 180, 30)

if st.button("Predict Calories"):
    input_data = np.array([[age, gender, height, weight, duration]])
    prediction = model.predict(input_data)[0]

    st.success(f"🔥 Calories Burned: {prediction:.2f}")

    # ---------------------------
    # LINE GRAPH 📈
    # ---------------------------
    durations = np.array([10, 20, 30, 40, 50, 60])
    calories = model.predict([
        [age, gender, height, weight, d] for d in durations
    ])

    plt.figure()
    plt.plot(durations, calories, marker='o')
    plt.title("Calories vs Duration")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Calories Burned")
    st.pyplot(plt)

    # ---------------------------
    # PIE CHART 
    # ---------------------------
    labels = ['Burned Calories', 'Remaining Energy']
    remaining = max(0, 1000 - prediction)  # assume 1000 calorie goal
    sizes = [prediction, remaining]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.set_title("Calories Distribution")

    st.pyplot(fig)