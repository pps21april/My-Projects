
import pandas as pd
import numpy as np
import random
import streamlit as st

# Defining the path of the dataset
file_path = 'dataset.csv'

# giving the sample size to be used for analysis
sample_size = 5000

# set a random seed for reproducibility
random.seed(42)

# Read the sample from the dataset
df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and random.random() > (sample_size / 30000),on_bad_lines='skip')

# filling missing values in Hotel Details with "Not Available"
df["Hotel Details"].fillna("Not Available", inplace = True)

# filling missing values in Airline with "Not Available"
df["Airline"].fillna("Not Available", inplace = True)

# filling missing values in Onwards Return Flight Time with "Not Available"
df["Airline"].fillna("Not Available", inplace = True)

# filling missing values in Sightseeing Places Covered  with "Not Available"
df["Sightseeing Places Covered"].fillna("Not Available", inplace = True)

# filling missing values in Cancellation Rules with "Not Available"
df["Cancellation Rules"].fillna("Not Available", inplace = True)

# dropping empty and unwanted columns
df.drop(columns = ["Flight Stops","Meals","Initial Payment For Booking","Date Change Rules","Crawl Timestamp","Company"], inplace = True)

# filtering unwanted package types
allowed_package_types = ["Deluxe","Standard","Premium","Luxury","Budget"]
df = df[df["Package Type"].isin(allowed_package_types)]

# converting travel date to datetime format
df["Travel Date"] = pd.to_datetime(df["Travel Date"], format = '%d-%m-%Y', errors='coerce')

# importing TfidfVectorizer class from scikit-learn library for converting text data into a matrix of numerical vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# importing linear_kernel class for finding cosine similarity between numerical values
from sklearn.metrics.pairwise import linear_kernel

# concatinating hotel details and destination
df["Hotel_Info"] = df["Hotel Details"].str.cat(df["Destination"], sep = "|")

# creating a TfidfVectorizer model
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# creating a matrix of numerical values by transforming hotel_info variable using TfidfVectorizer model
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Hotel_Info"])

# finding cosine similarity between rows of tfidf_matrix
cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a function for recommending hotels based on package_type,destination,price,start_city
def hotel_recommendation(package_type,start_city,price,destination,cosine_sim = cosine_sim):

  # creating a new dataframe having details of only the required hotels
  filtered_df = df[(df["Package Type"]==package_type) &
                   (df["Destination"]==destination) &
                   (df["Start City"]==start_city) &
                   (df["Price Per Two Persons"]<=price)]

  if filtered_df.empty:
    return "No Matching Hotels Found"

  # finding indexes of each row in filtered_df
  hotel_indices = filtered_df.index
  avg_similarity_score = []

  # calculating average similarity score for each hotel
  for idx in hotel_indices:
     score = sum(cosine_sim[idx])/len(cosine_sim[idx])
     avg_similarity_score.append(score)

  # creating a new dataframe having uniq id, hotel details and avg_similarity_score for each booking
  new_df = pd.DataFrame({'Unique_Id':filtered_df["Uniq Id"],
                         'Hotel_Details':filtered_df["Hotel Details"],
                         'Avg_Similarity_Score':avg_similarity_score})

  # sorting the new_df in descending order of avg_similarity_score
  recommended_hotels_df = new_df.sort_values(by="Avg_Similarity_Score",ascending = False)

  # returning the required hotel details
  return recommended_hotels_df[['Unique_Id','Hotel_Details']]

st.title('Hotel Recommendation App')

# Create dropdowns for Package Type, Start City, and Destination
package_types = df['Package Type'].unique()
start_cities = df['Start City'].unique()
destinations = df['Destination'].unique()

package_type = st.selectbox('Select Package Type:', package_types)
start_city = st.selectbox('Select Start City:', start_cities)
destination = st.selectbox('Select Destination:', destinations, format_func=lambda x: x.replace('|', ', '))

# Slider for Price
price = st.slider('Select Maximum Price:', min_value=0, max_value=df['Price Per Two Persons'].max(), value=10000)

# Get recommendations on button click
if st.button('Get Recommendations'):
    recommended_hotels = hotel_recommendation(package_type, start_city, price, destination)
    if isinstance(recommended_hotels, str):
        st.warning(recommended_hotels)
    else:
        st.table(recommended_hotels)

# Optionally, you can display some information about the selected filters
st.write('Selected Filters:')
st.write(f'Package Type: {package_type}')
st.write(f'Start City: {start_city}')
st.write(f'Destination: {destination}')
st.write(f'Maximum Price: {price}')