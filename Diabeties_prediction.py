import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

APP_ICON_URL = "DAPR Health.jpg"
st.set_page_config(
     page_title="DAPR HEALTH Diabetes Prediction",
     page_icon=APP_ICON_URL,
     layout="wide",)

t1, t2 = st.columns((1,10)) 
t1.image(APP_ICON_URL, width = 120)
t2.title("DAPR HEALTH Diabetes Prediction")

df= pd.read_csv(r"C:\Users\91995\OneDrive\Desktop\API\diabetes.csv")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)

def main():
    st.title("DAPR Health")
    st.subheader("Data")
    st.dataframe(X.describe())
    st.subheader("Model Performance")
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.subheader("Visual representation of the Data")
    st.line_chart(X)   
    st.subheader("Diabetes Prediction")
    st.write("Enter Your Data:")

    pregnancies = st.slider("Enter no. of Pregnencies", min_value=0, max_value=20)
    glucose = st.slider("Enter your Glucose Level", min_value=50, max_value=250)
    blood_pressure = st.slider("Enter your BloodPressure", min_value=40, max_value=150)
    skin_thickness = st.slider("Enter your SkinThickness", min_value=10, max_value=60)
    insulin = st.slider("Enter your Insulin Level", min_value=0, max_value=400)
    bmi = st.slider("Enter your BMI", min_value=10.0, max_value=50.0)
    diabetes_pedigree = st.slider("Enter your DiabetesPedigreeFunction", min_value=0.0, max_value=2.0, step=0.01)
    age = st.slider("Enter your age", min_value=20, max_value=80)

    if st.button("Submit"):
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })

        prediction = model.predict(input_data)
        if prediction[0] == 0:
            im=Image.open("safe.png")
            st.image(im)
        else:
            image =Image.open("have.png")
            st.image(image)
if __name__ == '__main__':
    main()

