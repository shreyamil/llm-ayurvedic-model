import json
import os
import streamlit as st
import requests

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
import datetime

from dotenv import load_dotenv
load_dotenv()

# Load the NVIDIA API key
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Vector embedding setup
def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Setting up the environment..."):
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFDirectoryLoader(r"data")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.set_page_config(page_title="Uttarakhand Traveling", layout="wide", page_icon="🏔")


# # Custom styling for the title
st.markdown("""
    <style>
        .title {
            font-size: 50px;  /* Increase the font size */
            font-weight: bold;  /* Make the font bold */
            color: #2E8B57;  /* Set text color */
            text-align: center;  /* Center the text */
            text-transform: uppercase;  /* Make the text uppercase */
            letter-spacing: 2px;  /* Add spacing between letters */
            padding: 20px;  /* Add padding for spacing around the title */
            background-color: #F0F8FF;  /* Set a light background color */
            border-radius: 10px;  /* Add rounded corners */
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);  /* Add a shadow effect */
            margin-bottom: 30px;  /* Add space below the title */
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Welcome to the Uttarakhand Trip Planner</div>', unsafe_allow_html=True)
st.image("utta.jpg", caption="Sunrise by the mountains")

# Horizontal card section with centered buttons
st.markdown('<div class="trip-section">', unsafe_allow_html=True)

# Creating horizontal columns for the cards
col1, col2, col3 = st.columns(3, gap="medium")

# CSS for card styling
st.markdown(
    """
    <style>
    .card {
        background-color: #F8F8F8;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        padding: 20px;
        text-align: center;
    }
    .card-title {
        font-size: 20px;
        font-weight: bold;
        color: #2E8B57;
        margin-bottom: 10px;
    }
    .card-image {
        border-radius: 8px;
        max-height: 150px;
        object-fit: cover;
        width: 100%;
    }
    .center-button {
        display: block;
        margin: 20px auto;
        padding: 10px 20px;
        background-color: #2E8B57;
        color: white;
        border: none;
        border-radius: 5px;
        text-decoration: none;
        font-size: 16px;
        cursor: pointer;
    }
    .center-button:hover {
        background-color: #3CB371;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Card 1 - Local Cuisine
with col1:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Local Cuisine</div>
            <img src="https://captureatrip-cms-storage.s3.ap-south-1.amazonaws.com/foods_of_uttarakhand_bfc32fef3b.webp" alt="Traditional Cuisine" class="card-image">
            <p>Discover the authentic flavors of Uttarakhand, Bhatt Ki Churdkani, and Baadi.</p>
            <button class="center-button">Learn More</button>
        </div>
        """,
        unsafe_allow_html=True
    )

# Card 2 - Popular Attractions
with col2:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Popular Attractions</div>
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlAjH7QK7eVtNDfjbcz8Ru8NlQi-9JjMWr6g&s">
            <p>Explore the scenic beauty of Nainital, Mussoorie, 
            Rishikesh, and Jim Corbett National Park.</p>
            <button class="center-button">Explore More</button>
        </div>
        """,
        unsafe_allow_html=True
    )

# Card 3 - Adventure Activities
with col3:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Adventure Activities</div>
            <img src="https://www.atlastravel.in/blog/wp-content/uploads/2019/05/Paragliding.jpg" alt="Adventure Activities" class="card-image">
            <p>Experience thrilling adventures like river rafting, 
            trekking, camping, and paragliding.</p>
            <button class="center-button">Discover More</button>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for trip planning input
val = 0
with st.sidebar:
    st.subheader("Plan Your Trip", divider="rainbow")
    
    # Input for destination
    prompt1 = st.text_input("Enter your Destination", help="Example: Nainital, Mussoorie")
    
    # Today's date
    today = datetime.date.today()
    # End of current year
    dec_31 = datetime.date(today.year, 12, 31)
    
    # Date input for vacation, starting from today
    vacation_dates = st.date_input(
        "Select your vacation dates",
        (today, today + datetime.timedelta(days=7)),  # Default range (1 week from today)
        min_value=today,  # Start from today
        max_value=dec_31,  # Till the end of the current year
        format="MM.DD.YYYY"
    )
    # Calculate number of days between the selected dates
    num_days = (vacation_dates[1] - vacation_dates[0]).days
    
    # Check if the number of days is more than 10
    if num_days > 10:
        st.error("⚠ Please enter a date range with 10 or fewer days.")
        prompt2 = None  # Set prompt2 to None if the validation fails
    else:
        prompt2 = num_days  # Set prompt2 as an integer number of days
    
    budget = st.slider("Select Your Budget (in INR)", 1000, 10000, step=500, value=5000, help="Select your budget for the trip")
    
    # Construct Google Maps URL based on the input destination
    maps = f"https://www.google.com/maps/place/{prompt1}"
    
    if st.button("Submit",type="primary"):
        val=1

if (val == 1):
    
    # Check the values of prompt1 and prompt2
    if prompt1 and prompt2:
        st.write(f"Destination: {prompt1}")
        st.write(f"Number of Days: {prompt2}")
        st.write(f"Selected Budget: ₹{budget}")

    vector_embedding()

# Generating response for the trip plan
    if not prompt1 or not prompt2:
        st.error("Please fill out both fields.")
    else:
        with st.spinner("Processing your request..."):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
            prompt = ChatPromptTemplate.from_template("""
            Provide a detailed trip plan for the given destination, number of days, and budget.
            Include recommendations for local cuisine, hotels, attractions, and activities within the specified budget.
            <context>{context}</context>
            Destination: {input}
            Days: {days}
            Budget: {budget} INR
            """)
        
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            try:
                response = retrieval_chain.invoke({'input': prompt1, 'days': prompt2, 'budget': budget})
                st.markdown(f"<div>{response['answer']}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred while processing your request: {str(e)}")

            processing_time = time.process_time() - start
        
            st.success(f"Response received in {processing_time:.2f} seconds")
            st.link_button("Click For Maps",maps,)
            binary_content = response['answer'].encode('utf-8')
        
            # Add a download button
            st.download_button(
            label="📄 Download Trip Plan as Text File",
            data=binary_content,
            file_name=f"{prompt1}.txt",
            mime="text/plain",
            )

    # Clear session state
    if st.button("Clear"):
        val = 0
        st.session_state.clear()


# # OpenWeatherMap API setup
# API_KEY = '6f1ea8b6b36d33639bfe33d7d5d4bad3'  # Replace with your OpenWeatherMap API key
# BASE_URL = "http://api.openweathermap.org/data/2.5/forecast?"

# def get_weather(city, start_date, end_date):
#     # Calculate the number of days in the trip
#     num_days = (end_date - start_date).days + 1
    
#     # API request URL
#     url = f"{BASE_URL}q={city}&cnt={num_days}&appid={API_KEY}&units=metric"
    
#     # Send the request to OpenWeatherMap API
#     response = requests.get(url)
#     data = response.json()

#     # Create a list to store weather data
#     weather_data = []

#     if response.status_code == 200:
#         for day in range(num_days):
#             date = start_date + datetime.timedelta(days=day)
#             temp = data['list'][day]['main']['temp']
#             weather = data['list'][day]['weather'][0]['description']
#             weather_data.append({
#                 "Date": date.strftime("%b %d, %Y"),
#                 "Temperature (°C)": temp,
#                 "Weather": weather.capitalize()
#             })
#     else:
#         st.error("Failed to get weather data.")

#     return weather_data

# # Input for destination and trip dates

# st.subheader("Plan Your Trip")
# city = st.text_input("Enter the destination", "Nainital")
    
# # Select start and end date for the trip
# today = datetime.date.today()
# end_date = st.date_input("Select end date", today + datetime.timedelta(days=7))

# # Fetch the weather data for the selected trip duration
# if city and end_date:
#     weather_info = get_weather(city, today, end_date)

# # Display weather information in a table
# if weather_info:
#     st.subheader("Weather Forecast for Your Trip")
#     weather_df = pd.DataFrame(weather_info)
#     st.table(weather_df)


# Path to the reviews file (change as needed)
reviews_file = "reviews.json"

# Function to load reviews from the JSON file
def load_reviews():
    if os.path.exists(reviews_file):
        try:
            with open(reviews_file, "r") as file:
                return json.load(file)  # Attempt to load the reviews
        except json.JSONDecodeError:
            # If JSON is invalid, print an error and return an empty list
            print(f"Error: The file '{reviews_file}' contains invalid JSON. Returning empty reviews.")
            return []
    else:
        # If the file doesn't exist, return an empty list
        return []

# Function to save a new review to the JSON file
def save_review(new_review):
    reviews = load_reviews()
    reviews.append(new_review)
    with open(reviews_file, "w") as file:
        json.dump(reviews, file)

# Initialize reviews list from the JSON file
reviews = load_reviews()

# Section for review submission
st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line to separate sections

# Review Submission Section
st.markdown('<div class="title">Submit Your Review</div>', unsafe_allow_html=True)

# Create the review submission form
with st.form("review_form"):
    name = st.text_input("Your Name", max_chars=100)
    rating = st.slider("Rating (1-5)", 1, 5, 5)
    review_text = st.text_area("Your Review", max_chars=500)
    
    # Submit button
    submit_button = st.form_submit_button("Submit Review")
    
    if submit_button:
        if name and review_text:
            new_review = {"name": name, "rating": rating, "review": review_text}
            save_review(new_review)
            st.success("Thank you for your review! 🎉")
        else:
            st.error("Please fill in all fields before submitting.")

# Display existing reviews in a horizontal row
st.markdown('<div class="title">Traveler Reviews</div>', unsafe_allow_html=True)

# Layout reviews in horizontal and vertical stack (3 reviews per row)
if reviews:
    # Create a row for the reviews
    for i in range(0, len(reviews), 3):
        cols = st.columns(3, gap="medium")  # Create 3 columns for each row
        for j in range(3):
            if i + j < len(reviews):  # Make sure we don't go out of bounds
                with cols[j]:
                    review = reviews[i + j]
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">{review['name']}</div>
                            <div class="card-rating">{"⭐" * review['rating']}</div>
                            <p>"{review['review']}"</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
else:
    st.write("No reviews yet. Be the first to submit your review!")

# Custom CSS for review cards styling
st.markdown("""
    <style>
        .review-card {
            margin-top: 20px;
            background-color: #F8F8F8;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .card-title {
            font-size: 18px;
            font-weight: bold;
            color: #2E8B57;
            margin-bottom: 10px;
        }
        .card-rating {
            color: gold;
            font-size: 16px;
            margin-bottom: 10px;
        }
        .card p {
            font-size: 14px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# Footer (Placed at the end of the page)
st.markdown("""
    <footer style="width:100%; background: linear-gradient(90deg, #2E8B57, #3CB371); text-align:center; padding:30px 0; color:white; border-top: 2px solid #fff; margin-top:40px;">
        <div style="max-width: 1200px; margin: 0 auto;">
            <p style="font-size: 18px; font-weight: bold;">© 2024 Uttarakhand Trip Planner</p>
            <p style="font-size: 14px;">All Rights Reserved | Designed with ❤ for travel enthusiasts</p>
        </div>
    </footer>
""", unsafe_allow_html=True)
