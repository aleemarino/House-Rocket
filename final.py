import pandas as pd
import streamlit as st
import geopandas
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime


st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
def extraction (path):
    data = pd.read_csv(path)
    return data

@st.cache(allow_output_mutation=True)
def change_to_time (data):
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    return data

@st.cache(allow_output_mutation=True)
def get_geodata(data):
    geofile = geopandas.read_file(url)
    return geofile


def set_features(data):
    data['m2_lot'] = data['sqft_lot'] / 0.092903
    data['price/m2'] = data['price']/data['m2_lot']
    data['level'] = data['price'].apply(lambda x: 'Level 01' if x <= 35000 else
                                                  'Level 02' if x <= 645000 else
                                                  'Level 03')
    data['dormitory_type'] = data['bedrooms'].apply(lambda x: 'Studio' if x == 1 else
                                                              'Apartment' if x == 2 else
                                                              'House')
    return data


def overview_data(data):
    filtered_data = data.select_dtypes(include=['int64', 'float64'])

    min_ = pd.DataFrame(filtered_data.apply(np.min, axis=0))
    max_ = pd.DataFrame(filtered_data.apply(np.max, axis=0))
    media = pd.DataFrame(filtered_data.apply(np.mean, axis=0))
    median = pd.DataFrame(filtered_data.apply(np.median, axis=0))
    std = pd.DataFrame(filtered_data.apply(np.std, axis=0))

    df_metrics = pd.concat([min_, max_, media, median, std], axis=1).reset_index()

    df_metrics.columns = ['Attribute', 'Min', 'Max', 'Media', 'Median', 'Std']

    data['m2_lot'] = data['sqft_lot'] * 0.0092903
    data['price/m2'] = data['price'] / data['m2_lot']

    df0 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df1 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df2 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['price/m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    m1 = pd.merge(df0, df1, on='zipcode', how='inner')
    m2 = pd.merge(m1, df2, on='zipcode', how='inner')
    m3 = pd.merge(m2, df3, on='zipcode', how='inner')

    f_attributes = st.sidebar.multiselect('Atributtes', data.columns)
    f_zipcode = st.sidebar.multiselect('Zipcode', data['zipcode'].unique())

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]
    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    else:
        data = data.copy()

    st.title('Data Overview')
    st.header('Data Sample')
    st.write(data.head(11))
    c1, c2 = st.beta_columns((1, 1))
    c1.header('Central Tendency & Dispersion Metrics')
    c1.write(df_metrics)
    c2.header('Informations by Zipcode')
    c2.write(m3)
    return None


def portfolio_density(data, geofile):
    st.title('Region Overview')

    c1, c2 = st.beta_columns((1, 1))
    c1.header('Portfólio Density')

    df = data.sample(20000)

    density_map = folium.Map(location=[data['lat'].mean(),
                                       data['long'].mean()],
                             default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)

    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold by ${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, {5} sqft_living, year built: {6}'.format(
                          row['price'],
                          row['date'],
                          row['sqft_lot'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['sqft_living'],
                          row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    # df = df.sample(20000)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()],
                                  default_zoom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlGnBu',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)
    return None


def commercial_distribution(data):
    st.title('Commercial Attributes')
    c1, c2 = st.beta_columns((1,1))
    # Average Price per year

    # filters

    st.sidebar.subheader('Select Max Year Built')

    f_year_built = st.sidebar.slider('Year Built', int(data['yr_built'].min()),
                                     int(data['yr_built'].max()),
                                     int(data['yr_built'].mean()))

    c1.header('Average Price Per Year Date')
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['price', 'yr_built']].groupby('yr_built').mean().reset_index()

    fig = px.line(df, x='yr_built', y='price')

    c1.plotly_chart(fig, use_container_width=True)

    # Average Price per Day

    st.sidebar.subheader('Select Max Day Built')

    c2.header('Average Price Per Day')

    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_day_built = st.sidebar.slider('Day Built', min_date,
                                    max_date,
                                    min_date)

    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_day_built]

    df = data[['price', 'date']].groupby('date').mean().reset_index()

    fig = px.line(df, x='date', y='price')

    c2.plotly_chart(fig, use_container_width=True)
    # Histogram

    # 1 - By price

    st.header('Price Distribution')

    f_histogram = st.sidebar.slider('Select Max Price to Build Histogram', int(data['price'].min()),
                                    int(data['price'].max()),
                                    int(data['price'].mean()))

    df_to_plot_graphic_3 = data.loc[data['price'] < f_histogram]

    map3 = px.histogram(df_to_plot_graphic_3, x='price', nbins=50)
    st.plotly_chart(map3, use_container_width=True)


    return None


def attributtes_distribution(data):
    ###### UNITS ATTRIBUTION ###############

    # 2 - Concentration by bedroom
    c1, c2 = st.beta_columns(2)
    c1.header('Concentration By Room')

    f_room = st.sidebar.selectbox('Select Minimum Bedrooms',
                                  sorted(set(data['bedrooms'].unique())))

    df = data.loc[data['bedrooms'] > f_room]

    map4 = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(map4, use_container_width=True)

    # 3 - Concentration by bathroom

    c2.header('Concentration By Bathroom')

    f_bathroom = st.sidebar.selectbox('Select Minimum Bathrooms',
                                      sorted(set(data['bathrooms'].unique())))

    df = data.loc[data['bathrooms'] > f_bathroom]

    map5 = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(map5, use_container_width=True)

    # 4 - Concentration by floor

    st.header('Concentration By Floor')

    f_floors = st.sidebar.selectbox('Select Minimum Floors',
                                    sorted(set(data['floors'].unique())))

    df = data.loc[data['floors'] >= f_floors]

    map6 = px.histogram(df, x='floors', nbins=19)
    st.plotly_chart(map6, use_container_width=True)

    # 4 - Concentration by Waterview

    st.header('Concentration By Waterview')
    f_waterfront = st.sidebar.checkbox('Waterview')

    if f_waterfront:
        data = data[data['waterfront'] == 1]
    else:
        data = data.copy()

    map7 = px.histogram(data, x='waterfront', nbins=19)
    st.plotly_chart(map7, use_container_width=True)
    return None

if __name__ == '__main__':
    #ETL
    #data extraction
    path = '/home/aleemarino/Desktop/repos/House-Rocket/datasets/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    data = extraction(path)
    geofile = get_geodata(url)
    #transformation
    data = set_features(data)
    data = change_to_time(data)
    overview_data(data)
    portfolio_density(data, geofile)
    commercial_distribution(data)
    attributtes_distribution(data)


















# Map plots



# Distribuição dos Imóveis por categorias comerciais




