from collections import namedtuple
import math
import pandas as pd
import streamlit as st
from geopy import geocoders
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from scipy.spatial import distance

"""
# Here is the test app for data visualization
"""

# Create geocoder object
# g = geocoders.GoogleV3() #api_key='Asefsefsefdsefsefsefs')

app = Nominatim(user_agent="tutorial")

# get pandas dataframe from file
# cloumn names: img_url, house_type, price, address, ad_url, square, object_type, latitude, longitude
realEstateDataFrame = pd.read_csv('data/rent_test.csv')

# Function for get geo coordinates from address
def get_location_by_address(address):
    try:
        #location = g.geocode(address, timeout=10)
        location = app.geocode(address, timeout=1000)
        latitude = location.latitude
        longitude = location.longitude
        return (float(latitude),float(longitude), location.address)
    except AttributeError:
        print("Problem with data or cannot Geocode.")
    except GeocoderTimedOut:
        print('GeocoderTimedOut')

# Show input for address
address = st.text_input(
        "Please type address here and press Enter:",
        "Espoo")
# Show select (st.selectbox) for type of property based on the list
# generated by realEstateDataFrame object_type column
# using 20 first rows of unique object_type 
# corting by count of rows with this type
# using descending order 
# showing object_type and count of rows with this type

object_type = st.selectbox(
        "Please select type of property:",
        realEstateDataFrame['object_type'].value_counts().head(20).index)

# get geo coordinates and address by address
lat, lon, adr = get_location_by_address(address)

st.write('The address you entered is:', adr)
st.write('The latitude is:', lat)
st.write('The longitude is:', lon)


# function to filter dataframe by distance from point (lat, lon) 
# using distance.euclidean function and return N nearest points
# filtered by object_type column before calculating distance

def get_nearest_points(df, lat, lon, N, filter_by_object_type=object_type):
    # filter dataframe by object_type
    df = df[df['object_type'] == filter_by_object_type]
    # calculate distance from point (lat, lon) to each point in dataframe
    df.loc[:, 'distance'] = df.apply(lambda row: distance.euclidean((lat, lon), (row['latitude'], row['longitude'])), axis=1)
    # sort dataframe by distance column in ascending order
    df = df.sort_values(by=['distance'])
    # return N nearest points
    return df.head(N)


# get 10 nearest points from dataframe and save it to new dataframe for st.map
# filtered by object_type column
df_for_map = get_nearest_points(realEstateDataFrame, lat, lon, 10)

# calculate average price for 10 nearest points
avg_price = df_for_map['price'].mean()

# display average price
st.write('Average price for 10 nearest points is:', avg_price)

# display map with markers based on the dataframe
st.map(df_for_map, zoom=11)

# with st.echo(code_location='below'):
#     total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#     num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

#     Point = namedtuple('Point', 'x y')
#     data = []

#     points_per_turn = total_points / num_turns

#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))

#     st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#         .mark_circle(color='#0068c9', opacity=0.5)
#         .encode(x='x:Q', y='y:Q'))
