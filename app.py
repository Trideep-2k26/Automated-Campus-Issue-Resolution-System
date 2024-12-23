import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Sample dataset
data = {
    "Division": ["Hostel", "Hostel", "Hostel", "Hostel", "Hostel",
                 "Academics", "Academics", "Academics", "Academics",
                 "Library", "Library", "Placement", "Placement", "Placement",
                 "Placement", "Mess/Cafeteria", "Mess/Cafeteria", "Mess/Cafeteria",
                 "Mess/Cafeteria", "Student Clubs", "Student Clubs", "sports"],
    "Problem ID": ["P001", "P002", "P003", "P004", "P005",
                   "P006", "P007", "P008", "P009",
                   "P010", "P011", "P012", "P013", "P014",
                   "P015", "P016", "P017", "P018",
                   "P019", "P020", "P021", "P022"],
    'Problem Statement': ['Room Maintenance Issue', 'Wi-Fi / Lan', 'Laundry Service Issue',
                          'Visitor Permission', 'Electricity Issues', 'Exam Schedule Issues',
                          'Grading or Marks Dispute', 'Course Registration Problem',
                          'Scholarship or Fee Waiver', 'Book Availability',
                          'Access or Fines', 'Internship Opportunities', 'Placement Training',
                          'Placement Registration', 'Company Availability', 'Cafeteria Hygiene',
                          'Meal Plan Changes', 'Food Quality Issues', 'Menu Suggestions',
                          'Club Registration', 'Event Information', 'sports related issues'],
    'Complaint Text': ['There’s a plumbing issue in my hostel room. Who can help with this?',
                       'The Wi-Fi in my hostel isn’t working. Who should I contact?',
                       'The hostel laundry service isn’t working. How do I report this?',
                       'How do I get permission for a visitor to stay overnight?',
                       'The power supply in my room is inconsistent. Whom should I contact?',
                       'When will the exam schedule be released?',
                       'I think there’s an error in my grades. How can I get it rechecked?',
                       'I am unable to register for my elective course. Who should I contact?',
                       'How do I apply for a scholarship or fee waiver?',
                       'I can’t find a specific book in the library. Can I request it?',
                       'I need help with paying my library fine. Who should I contact?',
                       'I need information on internships. Who can help me?',
                       'Will there be any training sessions for placements?',
                       'How do I register for campus placements?',
                       'Which companies will be visiting campus for placements?',
                       'The cafeteria needs better cleanliness. Who should I report to?',
                       'How can I switch my meal plan in the mess?',
                       'The food quality in the mess isn’t good. How can this be improved?',
                       'Can we get more variety in the mess menu?',
                       'How can I join a student club?', 'Is there an upcoming tech fest? How can I participate?',
                       'Gym sticker, badminton sticker'],
    'Contact Email': ['hostel.affairs@iiitdm.ac.in', 'cc-support@iiitdm.ac.in', 'hostel.affairs@iiitdm.ac.in',
                      'chief-warden@iiitdm.ac.in', 'hostel.affairs@iiitdm.ac.in', 'academic.affairs@iiitdm.ac.in',
                      'dean-ac@iiitdm.ac.in', 'academic.affairs@iiitdm.ac.in', 'pic-scholarships@iiitdm.ac.in',
                      'pic-library@iiitdm.ac.in', 'pic-library@iiitdm.ac.in', 'placement.affairs@iiitdm.ac.in',
                      'placement@iiitdm.ac.in', 'https://www.placements.iiitdm.ac.in/students', 'https://www.iiitdm.ac.in/placements',
                      'general.affairs@iiitdm.ac.in', 'https://docs.google.com/forms/d/e/1FAIpQLSewuVzcKSGdLsUdr6LoxkMUuZ9BXaaNGJhHw0EjZSBR6mn9QQ/viewform',
                      'mess.affairs@iiitdm.ac.in', 'https://docs.google.com/forms/d/e/1FAIpQLSewuVzcKSGdLsUdr6LoxkMUuZ9BXaaNGJhHw0EjZSBR6mn9QQ/viewform', 'technical.affairs@iiitdm.ac.in',
                      'technical.affairs@iiitdm.ac.in', 'sports-sac@iiitdm.ac.in / pic-sports@iiitdm.ac.in'],
    'Hours To Resolve': [48, 12, 72, 24, 48, 12, 24, 6, 24, 2, 4, 12, 24, 1, 1, 24, 1, 12, 1, 6, 6, 12]
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the TF-IDF Vectorizer on the complaint text
X = vectorizer.fit_transform(df['Complaint Text'])

# Initialize KNN with the number of neighbors set to 1 for simplicity
knn = NearestNeighbors(n_neighbors=1, metric='cosine')
knn.fit(X)

# Streamlit Interface
st.markdown("<style>body {background-color: #f5f5f5;}</style>", unsafe_allow_html=True)

# Dynamic Title
st.title("Campus Complaint Resolution Portal")

# Main Heading with Dynamic Content
input_method = st.radio("How would you like to describe your issue?", ("Type Complaint", "Select from Dropdown"), index=0, key="input_method", label_visibility="collapsed")

# Section for Dropdown Input Method
if input_method == "Select from Dropdown":
    st.subheader("Select your Division and Problem Statement to get in touch with the relevant contact")

    division = st.selectbox("Select your Division", df['Division'].unique(), key="division")

    filtered_df = df[df['Division'] == division]

    problem_id = st.selectbox("Select your Problem Statement", filtered_df['Problem Statement'], key="problem_statement")

    show_button = st.button("Get Contact Information", key="show_button")

    if show_button:
        selected_complaint = filtered_df[filtered_df['Problem Statement'] == problem_id].iloc[0]
        contact_info = selected_complaint['Contact Email']
        hours_to_resolve = selected_complaint['Hours To Resolve']

        if 'http' in contact_info:
            contact_display = f"**Click to go to the site/form**: [Visit the site]({contact_info})"
        else:
            contact_display = f"**Contact Email**: {contact_info}"

        st.markdown(contact_display)
        st.write(f"**Usually takes to resolve**: {hours_to_resolve} hours")

else:
    st.subheader("Type Your Complaint")
    typed_text = st.text_area("Type your Complaint Text", key="typed_text")
    show_button = st.button("Get Contact Information", key="show_button")

    if typed_text:
        typed_text_vector = vectorizer.transform([typed_text])
        distances, indices = knn.kneighbors(typed_text_vector)
        selected_complaint = df.iloc[indices[0][0]]

        contact_info = selected_complaint['Contact Email']
        hours_to_resolve = selected_complaint['Hours To Resolve']

        if 'http' in contact_info:
            contact_display = f"**Click to go to the site/form**: [Visit the site]({contact_info})"
        else:
            contact_display = f"**Contact Email**: {contact_info}"

        st.markdown(contact_display)
        st.write(f"**Usually takes to resolve**: {hours_to_resolve} hours")

# Footer Section: Brief About the Site
st.markdown("---")
st.subheader("About the Campus Complaint Resolution Portal")
st.markdown("""    
    The Campus Complaint Resolution Portal allows students to easily find the appropriate contacts for addressing campus issues. 
    Whether you have concerns related to hostels, academics, libraries, placements, or other campus facilities, this portal 
    helps you navigate through your problems and directly reach the right department or individual. 
    Get quick and effective resolution to your complaints!
""")
