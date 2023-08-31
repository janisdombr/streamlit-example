from collections import namedtuple
import math
import pandas as pd
import streamlit as st
from geopy import geocoders
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from scipy.spatial import distance
import folium
from streamlit_folium import folium_static
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# Function for get geo coordinates from address
def get_location_by_address(address):
    try:
        # Create geocoder object
        # g = geocoders.GoogleV3() #api_key='Asefsefsefdsefsefsefs')
        app = Nominatim(user_agent="tutorial")
        #location = g.geocode(address, timeout=10)
        location = app.geocode(address, timeout=1000)
        latitude = location.latitude
        longitude = location.longitude
        return (float(latitude),float(longitude), location.address)
    except AttributeError:
        print("Problem with data or cannot Geocode.")
    except GeocoderTimedOut:
        print('GeocoderTimedOut')

# function to filter dataframe by distance from point (lat, lon) 
# using distance.euclidean function and return N nearest points
# filtered by filter column before calculating distance

def get_nearest_points(df, lat, lon, N, filter='rooms', filter_value=2):
    # filter dataframe by object_type
    df2 = df[df[filter] == filter_value]
    
    # calculate distance from point (lat, lon) to each point in dataframe
    df2.loc[:, 'distance'] = df2.apply(lambda row: distance.euclidean((lat, lon), (row['latitude'], row['longitude'])), axis=1)

    # sort dataframe by distance column in ascending order
    df2 = df2.sort_values(by=['distance'])
    # return N nearest points
    return df2.head(N)

# function that returns random forest model trained on dataframe
# with columns: square, rooms, latitude, longitude, price
def get_random_forest_model(df):
    X = df[['square', 'rooms', 'latitude', 'longitude']].to_numpy()
    Y = df[['price']].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestRegressor().fit(X_train, y_train.ravel())
    test_predictions = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, test_predictions)
    # error = 1
    # for i in range(0, 5):
    #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #     model2 = RandomForestRegressor().fit(X_train, y_train.ravel())
    #     test_predictions = model2.predict(X_test)
    #     mape = mean_absolute_percentage_error(y_test, test_predictions)
    #     #st.write("MAPE: {0:7.2f} для модели случайного леса".format(mape))
    #     if mape < error:
    #         error = mape
    #         model = model2
    # #st.write("MAPE: {0:7.2f} минимальная, выбираем эту модель".format(error))
    return model, mape

st.header("Calculate your property price and payback with nearest points on map as examples")

col1, col2 = st.columns([1, 2])

# get pandas dataframe from file of selling objects
# cloumn names: img_url, house_type, price, address, ad_url, square, object_type, latitude, longitude
sell_data = pd.read_csv('data/sell_test.csv')

# get pandas dataframe from file of renting objects
# cloumn names: img_url, house_type, price, address, ad_url, square, object_type, latitude, longitude
# rent_data = pd.read_csv('data/rent_test.csv')

with col1:
    # Show input for address
    address = st.text_input(
            "Please type address:",
            "Espoo")
    # Show selectbox for number of rooms
    rooms = st.selectbox(
            "Please select number of rooms:",
            sell_data['rooms'].value_counts().head(20).index)
    # Train random forest model on filtered dataframe by rooms
    model, error = get_random_forest_model(sell_data[sell_data['rooms'] == rooms])
    # st.write('Database filtered : ', sell_data[sell_data['rooms'] == rooms].shape)

    # Show input for square
    square = st.text_input(
            "Please type square:",
            "50")
    
    # get geo coordinates and address by address
    lat, lon, adr = get_location_by_address(address)
    
    prediction = model.predict([[square, rooms, lat, lon]])
    
    # Show slider for number of rows to calculate average price
    # using 20 as default value
    # min value = 1 and max value = 100
    # step = 1 

    # range_slider = st.slider(
    #         "Please select number of nearest points:",
    #         1, 100, 20, 1)
    range_slider = 20
    
    # get (range_slider) nearest points from dataframe and save it to new dataframe for st.map
    # filtered by object_type column
    df_for_map = get_nearest_points(sell_data, lat, lon, range_slider, 'rooms', rooms)

    st.write('The address you entered is:', adr)
    # st.write('The latitude is:', lat)
    # st.write('The longitude is:', lon)

    # calculate average price for (range_slider) nearest points
    avg_price = df_for_map['price'].mean()
    # display average price
    st.write('Average price of ', range_slider, 'objects is: ', avg_price)

    st.write('Predicted price using ML is: ', prediction[0])

with col2:
    # display map with markers based on the dataframe
    # st.map(df_for_map, zoom=11, latitude=lat, longitude=lon)
    # use folium library to display map with markers containing html content
    m = folium.Map(location=[lat, lon], zoom_start=13)
    # iterate over dataframe rows and add marker with html content
    folium.Marker(
            [lat, lon],
            popup='Your position',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    for index, row in df_for_map.iterrows():
        pp= folium.Popup(folium.Html('<h3>'
                        + row['address']
                        + '</h3><br/><img src="'
                        + row['img_url']
                        + '" width="200px" /><br/> Price: '
                        + str(row['price'])
                        + '<br/> Rooms: '
                        + str(row['rooms'])
                        + '<br/> Square: '
                        + str(row['square'])
                        + '<br/><a href="'
                        + row['ad_url']
                        + '"target="_blank"> watch object </a>', script=True), max_width=250)
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=pp
        ).add_to(m)
    folium_static(m)
