import streamlit as st
import joblib
import pandas as pd

st.title("Parvoz sifatini bashorat qilish")

# Foydalanuvchidan ma'lumotlarni kiritishni so'rash
Gender = st.text_input("Jinsni kiriting(Male,Famale)")
Customer_Type = st.text_input("Xaridor turini kiriting(disloyal Customer,Loyal Customer)")
Age = st.number_input("Yosh", min_value=1, step=1)
Type_of_Travel = st.text_input("Sayohat turini kiriting(Business travel,Personal Travel)")
Class = st.text_input("Classni kiriting(Eco, Eco Plus, Business)")
Food_and_drink= st.number_input("Ovqat va ichimlik bahosini kiriting(1-5)", min_value=0,max_value=5, step=1)
Ease_of_Online_booking = st.number_input("Onlayn bronlashtirish osonligi darajasini kiriting (1-5)", min_value=1, max_value=5, step=1)
Seat_comfort = st.number_input("O'rindiq qulaylik darajasini kiriting(1-5)", min_value=0,max_value=5, step=1)
Inflight_entertainment = st.number_input("Parvoz ichidagi ko'ngilocharlik darajasi kiriting(1-5)", min_value=0,max_value=5, step=1)
Checkin_service = st.number_input("Ro'xatdan o'tish darajasini kiriting(1-5)", min_value=0,max_value=5, step=1)
Inflight_service = st.number_input("Parvoz ichidagi servis darajasini kiriting(1-5)", min_value=0,max_value=5, step=1)
Cleanliness = st.number_input("Tozalik darajasini kiriting(1-5)", min_value=0,max_value=5, step=1)

# Modelni yuklash va bashorat qilish
if st.button("Parvoz sifatini bashorat qilish"):
    # Kiritilgan ma'lumotlarni DataFrame ga o'tkazish
    input_data = {
        "Gender": [Gender],
        "Customer_Type": [Customer_Type],
        "Age": [Age],
        "Type_of_Travel": [Type_of_Travel],
        "Class": [Class],
        "Food_and_drink": [Food_and_drink],
        "Ease_of_Online_booking": [Ease_of_Online_booking],
        "Seat_comfort": [Seat_comfort],
        "Inflight_entertainment": [Inflight_entertainment],
        "Checkin_service": [Checkin_service],
        "Inflight_service": [Inflight_service],
        "Cleanliness": [Cleanliness]

    }
    
    df = pd.DataFrame(input_data)

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Customer_Type', 'Type_of_Travel', 'Class'])

    # Modelni yuklash
    model = joblib.load('decision_tree_model (3).pkl')  # Model faylingiz nomini mos ravishda kiriting

    # Bashorat qilish
    outcome = model.predict(df_encoded)

    # Natijani ko'rsatish
    st.write(f"Bashorat qilingan sifat: {outcome[0]}")