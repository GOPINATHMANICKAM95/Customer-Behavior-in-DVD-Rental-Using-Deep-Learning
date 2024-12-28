import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle as pkl

# Set the page configuration
st.set_page_config(page_title="Gopinath Entertainment Retails", layout="wide")

# Load models and encoders from pickle files
with open(r'Model\demand_model_encoders.pkl', 'rb') as file:
    encoders_app = pkl.load(file)

with open(r'Model\Demand_Model_Count.pkl', 'rb') as file:
    Demand_Model_Count = pkl.load(file)

with open(r'Model\Expected_Sales_Model.pkl', 'rb') as file:
    Expected_Sales_Model = pkl.load(file)

# Create two columns
col1, col2 = st.columns(2)

# Left column: Store ID, Date, and Prediction Table
with col1:
    st.title("Store Prediction App")    
    # Store ID selection
    store_ids = [1, 2]  # Add relevant IDs
    selected_store_id = st.selectbox("Select Store ID", store_ids)

    # Date selection with auto-fetched current date
    current_date = st.date_input("Select Date", datetime.now())    

    def pred(input_date, input_store_id):
        store_id = input_store_id
        date = input_date
        category_name_li = ['Documentary', 'Horror', 'Family', 'Foreign', 'Comedy', 
                            'Sports', 'Music', 'Animation', 'Action', 'New', 
                            'Sci-Fi', 'Classics', 'Games', 'Children', 'Travel', 'Drama']

        # Initialize the dictionary (lists) to store data temporarily
        pred = {"category": [], "count": []}
        df_data = {
            "rent_month": [],
            "rent_day": [],
            "rent_year": [],
            "category_name": [],
            "store_id": [],
            "rent_day_of_week": []
        }

        # Split the date into day, month, and year
        rent_day = date.day
        rent_month = date.month
        rent_year = date.year

        # Get the day of the week from the date (e.g., 'Tuesday')
        rent_day_of_week = date.strftime('%A')

        for category in category_name_li:
            # Append the data as a row to the dictionary
            df_data["rent_month"].append(rent_month)
            df_data["rent_day"].append(rent_day)
            df_data["rent_year"].append(rent_year)
            df_data["category_name"].append(category)
            df_data["store_id"].append(store_id)
            df_data["rent_day_of_week"].append(rent_day_of_week)

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(df_data)

        # Apply encoders to the appropriate columns
        for column in df.columns:
            if column in encoders_app:  # Check if an encoder exists for the column
                df[column] = encoders_app[column].transform(df[column])

        for i in range(len(category_name_li)):
            li = np.array(df.iloc[i].tolist()).reshape(1, -1)
            count = round(Demand_Model_Count.predict(li)[0], 0)
            pred["category"].append(category_name_li[i])
            pred["count"].append(count)            

        pred = pd.DataFrame(pred)
        count_li = pred["count"].to_list()
        df["rental_inventory_count"] = count_li
        
        li_1 = []
        for i in range(len(category_name_li)):
            li = np.array(df.iloc[i].tolist()).reshape(1, -1)
            amount = round(Expected_Sales_Model.predict(li)[0], 0)
            li_1.append(amount)

        pred["amount"] = li_1
        return pred

    # Generate predictions for the current date and the next two days
    pred1 = pred(current_date, selected_store_id)  # Predictions for today
    pred2 = pred(current_date + timedelta(days=1), selected_store_id)  # Tomorrow's predictions
    pred3 = pred(current_date + timedelta(days=2), selected_store_id)  # Day after tomorrow's predictions

    # Merge predictions into a single DataFrame for display
    prediction = pd.concat([pred1, pred2, pred3], axis=1, keys=["Day 1", "Day 2", "Day 3"])
    prediction = prediction.drop(columns=[('Day 2', 'category'), ('Day 3', 'category')])

    # Display the prediction demand table
    st.subheader("Prediction Demand Table")
    st.write(prediction)

# Right column: Visitor Inputs
with col2:
    # Load sales data and movie category cluster
    sales = pd.read_csv(r"Model\sales.csv")
    Movie_Category_Cluster = pd.read_csv(r"Model\Movie_Category_Cluster.csv")
    Actor = pd.read_csv(r"Model\Actor.csv")
    actor_analys = pd.read_csv((r"Model\actor_analys.csv"))

    def get_top_5_recommendations(customer_id, Movie_Category, sales):
        # Get the cluster for the given customer
        customer_cluster = Movie_Category.loc[Movie_Category['customer_id'] == customer_id, 'cluster'].values[0]

        # Filter customers in the same cluster
        same_cluster_customers = Movie_Category[Movie_Category['cluster'] == customer_cluster]['customer_id']

        # Aggregate movie preferences of the same cluster customers
        cluster_sales = sales[sales['customer_id'].isin(same_cluster_customers)]
        movie_counts = cluster_sales['category_name'].value_counts()

        # Get the top 5 recommended movies
        top_5_movies = movie_counts.head(5).index.tolist()

        return top_5_movies
    def get_top_5_actor(customer_id, Actor, actor_analys):
        # Step 1: Get the cluster for the given customer
        customer_cluster = Actor.loc[Actor['customer_id'] == customer_id, 'cluster'].values[0]

        # Step 2: Filter customers in the same cluster
        same_cluster_customers = Actor[Actor['cluster'] == customer_cluster]['customer_id']

        # Step 3: Aggregate movie preferences of the same cluster customers
        cluster_sales = actor_analys[actor_analys['customer_id'].isin(same_cluster_customers)]
        movie_counts = cluster_sales['actor_first_name'].value_counts()

        # Step 4: Get the top 5 recommended movies
        top_5_actors = movie_counts.head(5).index.tolist()

        return top_5_actors

    # Example usage:
  
    
    st.title("Visitor Information")    
    visitor_ids = sales["customer_id"].unique()  # Ensure unique IDs
    selected_visitor_id = st.selectbox("Select Visitor ID", visitor_ids)
   
    st.subheader("Fav Movie")
    top_5_movies = get_top_5_recommendations(selected_visitor_id, Movie_Category_Cluster, sales)
    st.write(top_5_movies)

    st.subheader("Fav Actor")
    # Replace this placeholder with actual data retrieval logic
    top_5_actors = get_top_5_actor(selected_visitor_id, Actor, actor_analys)
    st.write(top_5_actors)