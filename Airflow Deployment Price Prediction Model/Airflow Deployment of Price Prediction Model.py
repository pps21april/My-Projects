import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from airflow import DAG
from datetime import datetime,timedelta
from airflow.operators.python import PythonOperator


class DataLoader:
  def __init__(self,file_path):
      self.file_path = file_path

  def loader(self):
      df = pd.read_csv(self.file_path,on_bad_lines='skip')
      return df
  

class DataTransformer:
  def __init__(self,data):
      self.data = data

  def transformer(self):
      df = self.data

    # Dropping the columns with almost all values missing
      df.drop(['Date Change Rules', 'Initial Payment For Booking', 'Meals', 'Flight Stops'], axis=1, inplace=True)
    # Handling the missing values in the rest of the columns
      df['Hotel Details'].fillna('Not Available', inplace=True)
      df['Airline'].fillna('Not Available', inplace=True)
      df['Onwards Return Flight Time'].fillna('Not Available', inplace=True)
      df['Sightseeing Places Covered'].fillna('Not Available', inplace=True)
      df['Cancellation Rules'].fillna('Not Available', inplace=True)
    # Change travel date into a datetime object
      df['Travel Date'] = pd.to_datetime(df['Travel Date'], format='%d-%m-%Y', errors='coerce')
    # Remove the other values from package type
      allowed_types = ['Deluxe', 'Standard', 'Premium', 'Luxury', 'Budget']
      df = df[df['Package Type'].isin(allowed_types)]
    # Drop these irrelevant features
      df.drop(['Uniq Id', 'Crawl Timestamp', 'Page Url', 'Company'], axis=1, inplace=True)

    # extracting hotel ratings from Hotel Details column
      df['Hotel Ratings'] = df['Hotel Details'].str.extract(r'(\d+\.\d+)')
      df['Hotel Ratings'] = pd.to_numeric(df['Hotel Ratings'],errors='coerce')
      mode_rating = df['Hotel Ratings'].mode()[0]
      df['Hotel Ratings'].fillna(mode_rating,inplace=True)
    
    # encoding package_type and start_city
      df = pd.get_dummies(df,columns=["Package Type","Start City"])

    # removing outliers in per person price column
      q1 = df["Per Person Price"].quantile(0.25)
      q3 = df["Per Person Price"].quantile(0.75)

      IQR = q3-q1

      UB = q3 + 1.5*IQR
      LB = q1 - 1.5*IQR

      df = df[(df['Per Person Price']>=LB) & (df['Per Person Price']<=UB)]

    # extracting year, month and dayofweek from Travel Date
      df['Travel_Year'] = df['Travel Date'].dt.year
      df['Travel_Month'] = df['Travel Date'].dt.month
      df['Travel_DayofWeek'] = df['Travel Date'].dt.dayofweek

      return df     
  

class RandomForestModel:
   def __init__(self,data):
      self.data = data

   def random_forest(self):
      df = self.data

      # initialize the Sentence Transformer Model
      model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
      
    # Encode text-based columns and create embeddings
      text_columns = ['Package Name', 'Destination', 'Itinerary', 'Places Covered', 'Hotel Details', 'Airline', 'Sightseeing Places Covered', 'Cancellation Rules']  
      
      for column in text_columns:
          df[column + '_embedding'] = df[column].apply(lambda text : model.encode(text))

    # creating a PCA model to reduce the dimensions of embeddings
      n_components = 23
      pca = PCA(n_components=n_components)
    
    # reducing dimension of each embedding and creating new reduced embedded columns
      for column in text_columns:
          embedding = df[column + '_embedding'].values.tolist()
          
      
      numerical_features = ['Package Type_Standard', 'Package Type_Premium', 'Package Type_Luxury',
                            'Travel_Month', 'Package Type_Budget', 'Package Type_Deluxe',
                            'Hotel Ratings', 'Start City_New Delhi', 'Start City_Mumbai',
                            'Travel_DayOfWeek', 'Travel_Year']
      
      
      text_embeddings_pca = np.empty((len(df), n_components * len(text_columns)))

      for i, column in enumerate(text_columns):
          embeddings = df[column + '_embedding'].values.tolist()
          embeddings_pca = pca.fit_transform(embeddings)
          text_embeddings_pca[:, i * n_components:(i + 1) * n_components] = embeddings_pca
      
      X_numerical = df[numerical_features].values

      X = np.hstack((text_embeddings_pca,X_numerical))
      y = df['Per Person Price']

      X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      
    # scaling independent variables
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)
    
    # creating a random forest model 
      rf_model = RandomForestRegressor(n_estimators=100,max_depth=10,min_samples_split=2,random_state=42)
      
      rf_model.fit(X_train,y_train)
      y_pred = rf_model.predict(X_test)

      mae = mean_absolute_error(y_test,y_pred)
      mse = mean_squared_error(y_test,y_pred)
      rmse = np.sqrt(mse)
      r2 = r2_score(y_test,y_pred)

      pred = pd.DataFrame({'actual_value':y_test,'predicted_value':y_pred})
      print("The first five entries in pred dataframe is :\n",pred.head())

      
data_file_path = "https://airflowdemoliveclass.blob.core.windows.net/airflow/dags/dataset.csv"

load = DataLoader(data_file_path)
transform = DataTransformer(load.loader())
rf_model = RandomForestModel(transform.transformer())


default_args = {
   'owner':'admin',
   'depend_on_past': True,
   'start_date': datetime(2024,1,1),
   'retries' : 1,
   'retry_delay' : timedelta(minutes=1)
}

dag = DAG(
  'Travel_Price_Prediction',
  default_args=default_args,
  description = 'A Dag for travel price prediction',
  schedule_interval = None
)

load_data_task = PythonOperator(
   task_id = 'load_data_task',
   python_callable = load.loader,
   dag = dag
   )

transform_data_task = PythonOperator(
   task_id = 'transform_data_task',
   python_callable = transform.transformer,
   dag = dag,
   execution_timeout = timedelta(minutes = 30)
   )

random_forest_task = PythonOperator(
   task_id = 'random_forest_task',
   python_callable = rf_model.random_forest,
   dag = dag,
   execution_timeout = timedelta(minutes = 30)
   )

load_data_task >> transform_data_task >> random_forest_task










