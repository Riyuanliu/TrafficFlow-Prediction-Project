import folium
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
import geopandas as gpd
from shapely.wkt import loads
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from typing import Union, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



def clean_traffic_data(data):
    # Melt the DataFrame to unpivot the time columns
    melted_data = pd.melt(data, id_vars=['ID', 'SegmentID', 'Roadway Name', 'From', 'To', 'Direction', 'Date'],
                          var_name='TimeRange', value_name='Traffic Count')

    # Extract the start time from the TimeRange column using a more flexible approach
    melted_data['StartTime'] = melted_data['TimeRange'].str.extract(r'(\d+:\d+)', expand=False)

    # Convert the 'Date' and 'StartTime' columns to datetime type
    melted_data['Date'] = pd.to_datetime(melted_data['Date'])
    melted_data['StartTime'] = pd.to_datetime(melted_data['StartTime'], format='%H:%M', errors='coerce').dt.time

    # Convert 'StartTime' to string to use as column names
    melted_data['StartTime'] = melted_data['StartTime'].astype(str)

    # Extract the day of the week
    melted_data['DayOfWeek'] = melted_data['Date'].dt.day_name()

    # Pivot the table to create columns based on unique 'StartTime' values
    pivoted_data = melted_data.pivot_table(index=['Roadway Name', 'Date', 'DayOfWeek'],
                                           columns='StartTime', values='Traffic Count', aggfunc='mean').round(
        2).reset_index()

    # Flatten the multi-level column index
    pivoted_data.columns = [''.join(map(str, col)) for col in pivoted_data.columns]
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.title()
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('1 ', '1st ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('2 ', '2nd ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('3 ', '3rd ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('4 ', '4th ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('5 ', '5th ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('6 ', '6th ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('7 ', '7th ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('8 ', '8th ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('9 ', '9th ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('0 ', '0th ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('St ', 'st ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('Nd ', 'nd ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('Rd ', 'rd ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace('Th ', 'th ')
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace(r'\bAve$', 'Avenue', regex=True)
    pivoted_data['Roadway Name'] = pivoted_data['Roadway Name'].str.replace(r'\bSt$', 'Street', regex=True)
    pivoted_data = pivoted_data.sort_values(by=['Roadway Name', 'Date'])

    return pivoted_data


def clean_coordinates_data(df):
    # Convert the GeoJSON-like string to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=df['the_geom'].apply(loads))
    gdf = gdf.explode(index_parts=True)

    # Filter out points in a straight line and keep turning points
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: filter_turning_points(geom))

    # Extract the latitude and longitude for each point in the LineString
    gdf['Latitude'] = gdf['geometry'].apply(lambda g: g.xy[1].tolist())
    gdf['Longitude'] = gdf['geometry'].apply(lambda g: g.xy[0].tolist())

    # Keep the specified columns
    result = gdf[['Route_Name', 'Latitude', 'Longitude','Borough']]
    result = result.sort_values(by="Route_Name")
    result['Route_Name'] = result['Route_Name'].str.replace('1 ', '1st ')
    result['Route_Name'] = result['Route_Name'].str.replace('2 ', '2nd ')
    result['Route_Name'] = result['Route_Name'].str.replace('3 ', '3rd ')
    result['Route_Name'] = result['Route_Name'].str.replace('4 ', '4th ')
    result['Route_Name'] = result['Route_Name'].str.replace('5 ', '5th ')
    result['Route_Name'] = result['Route_Name'].str.replace('6 ', '6th ')
    result['Route_Name'] = result['Route_Name'].str.replace('7 ', '7th ')
    result['Route_Name'] = result['Route_Name'].str.replace('8 ', '8th ')
    result['Route_Name'] = result['Route_Name'].str.replace('9 ', '9th ')

    return result


def filter_turning_points(geom):
    if isinstance(geom, MultiLineString):
        # Filter turning points for each LineString in MultiLineString
        filtered_line_strings = [filter_turning_points(line) for line in geom]
        return MultiLineString(filtered_line_strings)
    elif isinstance(geom, LineString):
        # Filter turning points for the LineString
        coordinates = list(geom.coords)
        filtered_coordinates = [coordinates[0]]
        for i in range(1, len(coordinates) - 1):
            prev_point = Point(coordinates[i - 1])
            current_point = Point(coordinates[i])
            next_point = Point(coordinates[i + 1])
            if not is_in_straight_line(prev_point, current_point, next_point):
                filtered_coordinates.append(coordinates[i])
        filtered_coordinates.append(coordinates[-1])
        return LineString(filtered_coordinates)
    else:
        return geom


def is_in_straight_line(point1, point2, point3):
    # Check if three points are in a straight line
    line1 = LineString([point1, point2])
    line2 = LineString([point2, point3])
    return line1.parallel_offset(1).contains(line2)


def merge_dataframes(df1, df2, column1, column2):
    merged_data = pd.merge(df1, df2, left_on=column1, right_on=column2, how='inner')
    merged_data['month_of_year'] = pd.to_datetime(merged_data['Date']).dt.month
    return merged_data


def get_daily_traffic_mean(df):
    # Selecting relevant columns
    relevant_columns = ['DayOfWeek', 'Borough', '01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00', '06:00:00',
                        '07:00:00', '08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00']

    # Filtering and grouping by 'DayOfWeek' and 'Borough', then calculating mean traffic count
    daily_mean_traffic = df[relevant_columns].groupby(['DayOfWeek', 'Borough']).mean().reset_index()

    return daily_mean_traffic


def melt_traffic_data(df):
    # Selecting columns to melt
    id_vars = ['DayOfWeek', 'Borough']
    value_vars = ['01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00', '06:00:00',
                  '07:00:00', '08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00']

    # Melt the DataFrame
    melted_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Time', value_name='TrafficCount')

    return melted_df
def plot_streets_on_map(df, latitude_column, longitude_column):
    # Create a Folium map centered at the mean of the coordinates
    map_center = [43, -75]
    folium_map = folium.Map(location=map_center, zoom_start=14)

    # Iterate through the DataFrame rows and add each line to the map
    for index, row in df.iterrows():
        coordinates = list(zip(row[latitude_column], row[longitude_column]))
        folium.PolyLine(locations=coordinates, color='blue').add_to(folium_map)

    return folium_map


def plot_average_traffic_count_by_day(df, traffic_columns, day_of_week):
    # Filter the DataFrame based on the given day of the week
    day_df = df[df['DayOfWeek'] == day_of_week][['Route_Name'] + traffic_columns].copy()
    day_df = day_df.groupby('Route_Name').mean().reset_index()

    # Plot the average traffic count for each route
    plt.figure(figsize=(12, 6))
    for index, row in day_df.iterrows():
        plt.plot(traffic_columns, row[traffic_columns], label=row['Route_Name'])

    plt.title(f'Average Traffic Count on {day_of_week}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Traffic Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{day_of_week}_traffic_average.png')
    plt.close()


def split_test_train(df, x_col_names, y_col_name,test_size=25,random_state=2023) -> Union[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = df[x_col_names]
    y = df[y_col_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def calculate_average_traffic_by_day(data_df, time_columns):
    # Create a new DataFrame to store the results
    relevant_columns = ['Roadway Name', 'DayOfWeek'] + time_columns

    # Group by StreetName and DayOfWeek, and calculate the average traffic count for each time frame
    grouped_df = data_df[relevant_columns].groupby(['Roadway Name', 'DayOfWeek']).mean().round(2).reset_index()
    grouped_df["Latitude"] = data_df["Latitude"]
    grouped_df["Longitude"] = data_df["Longitude"]

    return grouped_df


def split_test_train(df, x_col_names, y_col_name, test_size=0.25, random_state=2023) -> Union[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """

    :type y_col_name: object
    """
    x = df[x_col_names]
    y = df[y_col_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def predict_traffic_count(x_train, x_test, y_train, y_test):
    # Assuming categorical_columns contains the names of columns with categorical data
    # Encode categorical columns in both x_train and x_test
    x_train_encoded = pd.get_dummies(x_train, columns='DayOfWeek')
    x_test_encoded = pd.get_dummies(x_test, columns='DayOfWeek')

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(x_train_encoded, y_train)

    # Predict on the test data
    y_pred = model.predict(x_test_encoded)

    # Create a DataFrame to store the actual and predicted traffic counts
    result_df = pd.DataFrame({'ActualTraffic': y_test, 'PredictedTraffic': y_pred})

    return result_df


def plot_model_accuracy(df):
    y_true = df['ActualTraffic']
    y_pred = df['PredictedTraffic']
    # Plot the true vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='red', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('True Traffic Count')
    plt.ylabel('Predicted Traffic Count')
    plt.grid(True)
    plt.show()


def main():
    file_path = 'Traffic_Volume_Counts_20231109.csv'
    # read the file into traffic data
    traffic_data = pd.read_csv(file_path)
    # clean the data
    cleaned_data = clean_traffic_data(traffic_data)
    # clean the location data
    location_data_path = 'DCM_ArterialsMajorStreets.csv'
    location_data = pd.read_csv(location_data_path)
    location_data = clean_coordinates_data(location_data)
    # merge the data
    merged_df = merge_dataframes(cleaned_data, location_data, 'Roadway Name', 'Route_Name')
    mean_merged_df = get_daily_traffic_mean(merged_df)
    mean_merged_melt = melt_traffic_data(mean_merged_df)
    location_data.to_csv("cleaned_location_data.csv", index=False)
    cleaned_data.to_csv("cleaned_traffic_data.csv", index=False)
    merged_df.to_csv("merged_data.csv", index=False)
    mean_merged_df.to_csv("mean_merged.csv", index=False)
    mean_merged_melt.to_csv("mean_merged_melt.csv", index=False)
    # map the street onto folium
    myapp = plot_streets_on_map(merged_df, 'Latitude', 'Longitude')
    myapp.save("route_map.html")  # Save the map as an HTML file (optional)
    #create a new df that predict the traffic based on time of day, day of week, and month of year
    time_columns = ['01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00', '06:00:00', '07:00:00', '08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00']
    # plot_average_traffic_count_by_day(merged_df, time_columns, 'Monday')
    # plot_average_traffic_count_by_day(merged_df, time_columns, 'Tuesday')
    # plot_average_traffic_count_by_day(merged_df, time_columns, 'Wednesday')
    # plot_average_traffic_count_by_day(merged_df, time_columns, 'Thursday')
    # plot_average_traffic_count_by_day(merged_df, time_columns, 'Friday')
    # plot_average_traffic_count_by_day(merged_df, time_columns, 'Saturday')
    # plot_average_traffic_count_by_day(merged_df, time_columns, 'Sunday')
    predict_df = calculate_average_traffic_by_day(merged_df,time_columns)
    x_train, x_test, y_train, y_test = split_test_train(mean_merged_melt, 'Time', 'TrafficCount')
    # Use the predict_traffic_count function
    result_df = predict_traffic_count(x_train, x_test, y_train, y_test)
    plot_model_accuracy(result_df)
    result_df.to_csv("predict_data.csv", index=False)



if __name__ == "__main__":
    main()
