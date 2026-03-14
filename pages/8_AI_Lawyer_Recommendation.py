import streamlit as st

st.title("AI Legal Advisor")

legal_issue = st.selectbox(
"Select Legal Issue",
[
"Family Dispute",
"Contract Issue",
"Business Registration",
"Property Dispute",
"Criminal Defense"
]
)

urgency = st.slider("Urgency Level",1,5,3)

if st.button("Get Lawyer Recommendation"):

    if legal_issue == "Family Dispute":
        lawyer = "Family Law Specialist"

    elif legal_issue == "Contract Issue":
        lawyer = "Corporate Lawyer"

    elif legal_issue == "Business Registration":
        lawyer = "Business Law Consultant"

    elif legal_issue == "Property Dispute":
        lawyer = "Property Law Expert"

    else:
        lawyer = "Criminal Defense Attorney"

    if urgency >=4:
        service="Immediate consultation recommended"
    else:
        service="Standard consultation"

    st.success(f"Recommended Lawyer: {lawyer}")

    st.write(f"Service Priority: {service}")

    st.info("""
Insight

AI recommendation helps users quickly find the right legal professional.
This reduces search friction and increases conversion rates.
""")
