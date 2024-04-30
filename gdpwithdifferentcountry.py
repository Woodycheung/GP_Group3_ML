"""
@author:
CHAN Chung Hang  22061759S
CHEUNG Ho Bun  22056983S
POON Wing Fung 22056100S
YEUNG Ka Wai 22049550S
"""


import pandas as pd
import commonFunctions as CF

# Load the GDP and visitor data
gdp_data = pd.read_csv('Data_GDP.csv')
visitor_data = pd.read_csv('Data_Visitor.csv')

# Cleaning and transforming GDP data
gdp_years = gdp_data.iloc[1, 2:].astype(str)  # Get the years
gdp_quarters = gdp_data.iloc[2, 2:].astype(str).str.extract(r'(\d)').astype(int)  # Extract the quarters

# Create the cleaned GDP DataFrame
gdp_cleaned = pd.DataFrame({
    'Year':gdp_years.values,
    'Quarter': gdp_quarters[0],  # The extracted quarters
    'GDP_Million_HKD': gdp_data.iloc[4, 2:].values# The GDP values starting from the fifth row
})

# Convert 'Year' and 'Quarter' to integers
gdp_cleaned['Year'] = gdp_cleaned['Year'].astype(int)
gdp_cleaned['Quarter'] = gdp_cleaned['Quarter'].astype(int)

# Prepare visitor data by different countries
new_visitor_data = pd.read_csv("Data_Visitor.csv", skiprows=4)
new_visitor_data.columns = ['Year', 'Month', 'Africa', 'Americas', 'Australia_NewZealand_SouthPacific', 'Europe',
                            'MiddleEast', 'NorthAsia', 'SouthAsia_SoutheastAsia', 'MainlandChina', 'Taiwan', 'Macau',
                            'Unidentified', 'Total']

# Clean and prepare visitor data
new_visitor_data['Year'] = new_visitor_data['Year'].replace('\xa0', '').str.strip()
new_visitor_data['Month'] = new_visitor_data['Month'].replace('\xa0', '').str.strip()
new_visitor_data = new_visitor_data[(new_visitor_data['Year'].str.isnumeric()) & (new_visitor_data['Month'].str.isnumeric())]
new_visitor_data['Year'] = new_visitor_data['Year'].astype(int)
new_visitor_data['Month'] = new_visitor_data['Month'].astype(int)


# Convert month to quarter
new_visitor_data['Quarter'] = new_visitor_data['Month'].apply(CF.get_quarter)

# Group by year and quarter
quarterly_data = new_visitor_data.groupby(['Year', 'Quarter']).sum().reset_index()

# print(quarterly_data)
# print(gdp_cleaned)

# Merge the GDP and visitor data on 'Year' and 'Quarter'
merged_data = pd.merge(gdp_cleaned, quarterly_data, on=['Year', 'Quarter'], how='inner')

# Analysis with before and after 2019
data_before_2019 = merged_data[merged_data['Year'] <= 2019]
data_after_2019 = merged_data[merged_data['Year'] >= 2019]
# print(merged_data)


print("All Data from 2002 to 2023")
CF.analysis_plot_result(merged_data,"Data_2002_2023")
print("Data Before 2019")
CF.analysis_plot_result(data_before_2019,"Data_Before_2019")
print("Data After 2019")
CF.analysis_plot_result(data_after_2019,"Data_after_2019")



"""

Reference List :

Hong Kong Special Administrative Region, Census and Statistics Department. (n.d.). Hong Kong GDP data. 
Retrieved from https://www.censtatd.gov.hk/tc/web_table.html?id=31

Hong Kong Special Administrative Region, Census and Statistics Department. (n.d.). Hong Kong visitor data. 
Retrieved from https://www.censtatd.gov.hk/tc/web_table.html?id=650-80001

NVIDIA. (n.d.). Linear Regression vs. Logistic Regression. 
Retrieved from https://www.nvidia.cn/glossary/data-science/linear-regression-logistic-regression/

NVDIA. (n.d.). Random Forest Regression
Retrieved from https://www.nvidia.cn/glossary/data-science/random-forest/

"""