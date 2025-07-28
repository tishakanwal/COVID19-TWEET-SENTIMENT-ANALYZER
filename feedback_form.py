import streamlit as st

st.title("📝 Feedback Form")

# Input fields
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, max_value=120)
email = st.text_input("Enter your email")

# Rating
rating = st.slider("How would you rate our app?", 1, 10, 5)

# Submit button
if st.button("Submit"):
    if name and email:
        st.success("🎉 Thank you for your feedback!")
        st.markdown("### 📋 Your Input Summary:")
        st.write(f"**Name:** {name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Email:** {email}")
        st.write(f"**Rating:** {rating}/10")
    else:
        st.warning("⚠️ Please fill in all required fields (Name and Email).")
