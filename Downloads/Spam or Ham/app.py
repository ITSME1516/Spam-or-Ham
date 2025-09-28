import streamlit as st
import pickle 

label = { 0: 'ham',  1 : 'spam'}

st.header(":red[Spam or Ham Detection Using Machine Learning]",divider=True)
st.write("""
Spam or Ham Detection is a common text classification problem in Natural Language Processing (NLP) aimed at automatically distinguishing between spam and legitimate (ham) messages. This project leverages machine learning algorithms to build a predictive model that can effectively identify and classify messages as spam or ham.

The key steps involved include data collection, preprocessing, feature engineering, model building, evaluation, hyperparameter tuning, and deployment. Technologies such as Python, Scikit-learn, NLTK, and Streamlit will be used to develop and deploy the model. The final deliverables include a cleaned dataset, a trained model, a web application for real-time predictions, and comprehensive documentation.

This project has applications in email filtering, SMS filtering, and content moderation on online platforms. Future enhancements may include model improvement using deep learning techniques, multilingual support, and continuous learning mechanisms.
""")

with open(r"Diployment\transformer.pkl","rb") as file:
    transformer = pickle.load(file)

with open(r"Diployment\model.pkl","rb") as file:
    model = pickle.load(file)

st.divider()

msg = st.text_area("Enter the Message:",placeholder="Hello test your sms or mail")
c1, c2, c3 = st.columns([4,2,4])
if c2.button("Predict",type="primary"):
    msg = transformer.transform([msg])
    prect = model.predict(msg)
    st.write(f"## The above message is :{'green' if label[prect[0]] == 0 else 'red'}[{label[prect[0]]}]")



