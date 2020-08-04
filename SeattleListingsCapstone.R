# R Code for the HarvardX PH125.9x Capstone CYO Project

# Written and prepared by Christian McKinnon

###############################################################################

# Note:
# A detailed introduction, sections on methods & analysis, modeling
# explanations and a conclusion can be found in the RMD and PDF reports.
# This is simply the code written to generate the figures in those reports!

# Encoding:
# Due to the presence of Chinese characters in the original dataset, this .R 
# file has been saved with UTF-8 encoding.

# Warnings:
# Warnings should be limited to messages from pacman installation packages and 
# "set.seed(123, sample.kind = "Rounding")" messages as this file was created
# using R 3.6.

###############################################################################

### Executive Summary ###

# It was back in October of 2007 that Joe Gebbia and Brian Chesky decided to throw 
# an air mattress in their living room and set up an "air bed and breakfast" for 
# guests arriving in San Francisco for a major convention. From this light-bulb 
# moment, the founders managed to attract investment and expand their vision to 
# a global empire with a 2020 private valuation of USD $26 billion. Travelers 
# now have the option of staying at short-term vacation rentals or "airbnbs" 
# in many parts of the world, creating a particularly efficient market for hosts 
# who need their listings to be priced competitively. Major cosmopolitan cities 
# like Seattle approaching nearly 10,000 listings will certainly create a demand 
# for accurate price prediction and forecasting.

# Our analysis will highlight specific elements of the "Seattle Airbnb Listing" 
# Dataset from 2018 which features 7,576 listings with 18 variables, scraped and 
# maintained by Tom Slee, and available for download on 
# [Kaggle](https://www.kaggle.com/shanelev/seattle-airbnb-listings). 

# In this report for the the Capstone course of the HarvardX Professional 
# Certificate in Data Science (PH125.9x), we will begin by examining the 
# unrefined data and perform any necessary tidying of the dataset. We will then 
# determine which features provide most insight into price prediction and remove 
# those that only serve to add noise. Data visualization and exploratory data 
# analysis will then be performed on the tidied dataset to inform our modeling. 
# Several machine learning algorithms will be applied to improve our RMSE (the 
# root mean square error) from our baseline to our final model.


### Methods and Analysis ###
# First we begin with data preparation and loading the data from Kaggle. After 
# the data has been loaded successfully, we will proceed to clean the data by 
# checking it for NAs and other errors such as misspellings or incorrect 
# categorizations or associations.

# Once the NAs have been removed and the data tidied, we proceed to our 
# exploratory data analysis where we examine correlations between the price per 
# night and the other features. Visualizations will illuminate these connections 
# and confirm our hypotheses.

# Modeling will involve the partitioning of our dataset "airbnb" into training, 
# validation, and test sets. In this report, the "validation" set will be used 
# in conjunction with the training set. This is in contrast to the terminology 
# used in the MovieLens Project, but more in line with contemporary machine 
# learning conventions. We will start with a baseline model based on the median 
# price of the dataset and move on to more advanced algorithms including linear 
# models, elastic net regression, regression trees, random forest models, kNN, 
# and neural nets all tuning with the training set "airbnb_train". Models will 
# be evaluated based on which produces the lowest RMSE. 

# Mathematically, the RMSE is the standard deviation of the prediction errors 
# (residuals) and is used to measure the difference between observed and predicted 
# values. The advantage of using the RMSE is that its unit is the same as the unit 
# being measured (in this case the price in USD). The model that produces the 
# lowest RMSE on the validation set will then be run on the test set.

### Data Preparation and Required Packages ###

# We will begin by loading the following libraries: tidyverse, readr, data.table,
# icesTAF, caret, lubridate, glmnet, scales, stringr, dplyr, ggmap, ggcorrplot,
# treemapify, rpart, nnet, formatR, rmarkdown, and knitr with the "pacman" package. 
# (If a package below is missing, p_load will automatically download it from CRAN).

if(!require(pacman)) install.packages("pacman", repos = "http://cran.us.r-project.org")
library(pacman)
pacman::p_load(tidyverse, readr, data.table, icesTAF, caret, lubridate, 
               ggthemes, ggplot2, glmnet, scales, stringr, dplyr, ggmap, ggcorrplot, 
               treemapify, rpart, nnet, formatR, rmarkdown, knitr)

### Data Preparation ###
# Download the Dataset:
if(!dir.exists("SAirbnb")) mkdir("SAirbnb")
if(!file.exists("./SAirbnb/seattle_01.csv")) download.file("https://raw.githubusercontent.com/christianmckinnon/Seattle-Airbnb-Listings/master/seattle_01.csv", "./SAirbnb/seattle_01.csv")

# Read the Data:
suppressWarnings(airbnb <-read_csv("./SAirbnb/seattle_01.csv"))
# Set the number of significant digits to 4
options(digits = 4)

### Preliminary Data Exploration and Cleaning ###

# Check dimensions of dataset
dim(airbnb)
# There are 7576 observations and 18 features

# Check the classes of the features: 
str(airbnb)
# We note that there 18 features of numeric, character, and POSIXct classes. We
# also notice that there is already one visible NA value in the
# "overall_satisfaction" feature.

### Dating Cleaning ###

# Coerce airbnb tibble into a data frame:
airbnb <-as.data.frame(airbnb)
class(airbnb)

# Confirm that the data is tidy and that there are no NAs:
sum(is.na(airbnb))
# There are 1475 NAs in the dataset.

# Check which features have NAs:
colSums(is.na(airbnb))

# We find that "overall_satisfaction" has 1473 NAs while bathrooms has 2.

# Create a dataframe na_bar to plot the NAs:
na_vis <- data.frame(t(colSums(is.na(airbnb))))
na_bar <- data.frame(Features = names(na_vis),totals=colSums(na_vis))

# Let's observe the NA distribution visually:
na_bar %>% ggplot(aes(x = reorder(Features, totals), y = totals, fill = Features, label = totals))+
  geom_bar(stat = "identity")+
  ggtitle("NA Distribution")+
  xlab("Features")+
  ylab("Total NAs")+
  coord_flip()+
  geom_text(size = 3, position = position_stack(vjust = 0.5))+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))
# The visualization confirms there are only two features with missing values
# and that the vast majority of observations are present.

# After confirming that there are only 2 features with NAs, (overall_satisfaction 
# & bathrooms) we proceed to clean the dataset.
# As overall_satisfaction has 1473 NAs, assigning a value of 0 will significantly
# skew the ratings negatively. Therefore we will fill the values with the mean
# to provide for more accurate predictive values.

# Convert NAs to the mean value:
airbnb$overall_satisfaction[is.na(airbnb$overall_satisfaction)] <- mean(airbnb$overall_satisfaction, na.rm = TRUE)
# The mean is roughly 4.84.
mean(airbnb$overall_satisfaction)

# Confirm the absence of NAs in this feature:
head(airbnb$overall_satisfaction)

# For the bathrooms feature, there are only 2 NAs and so we set them to zero.
airbnb <-airbnb %>% replace_na(list(bathrooms = 0))

# Now confirm the absence of any NAs in the dataset:
sum(is.na(airbnb))

### Feature Exploration and Selection ###

names(airbnb)
# There are 18 features and those less related to price prediction will be 
# dropped to refine our EDA and Modeling Focus:

# Feature: "X1"
head(airbnb$X1)
# "X1" is simply a numerical list for the dataset.

# Features: "room_id" & "host_id"
head(airbnb$room_id)
head(airbnb$host_id)
# "room_id" & "host_id" are arbitrary numbers assigned to identify rooms and hosts.

# Feature: "address"
# Check for unique values of the feature "address"
airbnb %>% select(address) %>% distinct()
# Note that there are 27 values with different formats and 12 repeated 
# instances of "Seattle."

# Any neighborhood of Seattle, the Chinese language version of "Seattle" and
# listings with only the State of Washington, will all be converted to Seattle.
# "WA" and "United States" will also be removed as they are redundant.
address_clean <-gsub("Seattle, WA, United States", "Seattle",
  gsub("Kirkland, WA, United States", "Kirkland",
  gsub("Bellevue, WA, United States", "Bellevue",
  gsub("Redmond, WA, United States", "Redmond",
  gsub("Mercer Island, WA, United States", "Mercer Island",
  gsub("Seattle, WA", "Seattle",
  gsub("Renton, WA, United States", "Renton",
  gsub("Ballard, Seattle, WA, United States", "Seattle",
  gsub("West Seattle, WA, United States", "Seattle",
  gsub("Medina, WA, United States", "Medina",
  gsub("Newcastle, WA, United States", "Newcastle",
  gsub("Seattle , WA, United States", "Seattle",
  gsub("Ballard Seattle, WA, United States", "Seattle",
  gsub("Yarrow Point, WA, United States", "Yarrow Point",
  gsub("Clyde Hill, WA, United States", "Clyde Hill",
  gsub("Tukwila, WA, United States", "Tukwila",
  gsub("Seattle, Washington, US, WA, United States", "Seattle",
  gsub("Capitol Hill, Seattle, WA, United States", "Seattle",
  gsub("Kirkland , Wa, United States", "Kirkland",
  gsub("Hunts Point, WA, United States", "Hunts Point",
  gsub("Seattle, DC, United States", "Seattle",
  gsub("Seattle, United States", "Seattle",
  gsub("Vashon, WA, United States", "Vashon",
  gsub("Kirkland , WA, United States", "Kirkland",
  gsub("Bothell, WA, United States", "Bothell",
  gsub("Washington, WA, United States", "Seattle",
      airbnb$address))))))))))))))))))))))))))

# Replace the Chinese version of "Seattle" separately using regex:
address_clean2 <-gsub(".*WA*.", "Seattle", address_clean)

# Reassign the column to the feature "address"
airbnb$address <-gsub("Seattle, United States", "Seattle", 
                 gsub("Seattle United States", "Seattle", address_clean2))

# Now confirm there are only 14 different cities and sort by the greatest
# numbers of listings:
city_list <-airbnb %>% group_by(address) %>% summarize(listing_sum = n()) %>%
  arrange(-listing_sum)
city_list
# It is clear the vast majority of listings are in Seattle (6791)

# Let's explore a data visualization to confirm this:
city_list %>% 
  ggplot(aes(x = reorder(address, listing_sum), y = listing_sum, 
                         fill = address, label = listing_sum))+
                         geom_bar(stat = "identity")+
                         ggtitle("Location Distribution")+
                         xlab("Location")+
                         ylab("Total Listings")+
                         coord_flip()+
                         geom_text(size = 4, 
                                   position = position_stack(vjust = 0.5))+
                         theme_bw()+
                         theme(plot.title = element_text(hjust = 0.5))
# Note: As the remaining locations are all cities, the feature "address" will later
# be renamed "city."
# Feature: "last_modified"
# The "last_modified" feature refers to the date a listing was updated and 
# nearly all values occur on 2018-12-20, telling us very little about the data,
# therefore this feature will be removed.
head(airbnb$last_modified)

# Feature: "location"
# The "location" feature will be removed in favor of using "latitude" & "longitude."
head(airbnb$location)

# Feature: "name"
# The "name" feature will be removed as it is a categorical description of each listing.
head(airbnb$name)

# Feature: "currency"
# The "currency" feature will be dropped as all rates are in US Dollars.
airbnb %>% select(currency) %>% distinct()

# Feature: "rate_type"
# Check for distinct values of "rate_type":
airbnb %>% select(rate_type) %>% distinct()
# After confirming only one unique value, "nightly," we determine this feature can be removed.

### Create the cleaned dataset ###
# Remove the above mentioned features and rename the columns:
airbnb <-airbnb %>% select(-c(X1, room_id, host_id, last_modified,
                              location, name, currency, rate_type)) %>%
                    rename(city = address, rating = overall_satisfaction,
                           reviews_sum = reviews)
# Reorder the columns:
airbnb <-airbnb[,c(8, 2, 4, 3, 1, 6, 7, 5, 9, 10)]

# Confirm the features have been tidied and reordered with only 10 features:
names(airbnb)

# Check the first few values of the cleaned dataset:
head(airbnb)

### Explanatory Data Analysis ###

# Now we will begin analyzing the features of our dataset to inform our 
# modeling approach.

# Correlogram:

# Remove non-numeric features:
airbnb_num <-airbnb %>% select(-c(city, room_type)) 

# Create the correlation matrix:
airbnb_cor <-cor(airbnb_num)

# Plot the Correlogram
ggcorrplot(airbnb_cor)+
  labs(title = "Airbnb Correlogram")+
  theme(plot.title = element_text(hjust = 0.5))
# We discover the price is moderately correlated with the number of people a 
# listing can accommodate as well as the number of bedrooms and bathrooms.
# Surprisingly, there is little correlation between the location (latitude &
# longitude) and price. This relationship will be explored with a scatterplot.

# Density Plot of Price Distribution below $300:
airbnb %>% filter(price <=300) %>% ggplot(aes(price))+
  geom_density(fill = "deepskyblue", size = 1.5, color = "navyblue", alpha = 0.5)+
  xlab("Price")+
  ylab("Density")+
  ggtitle("Price Distribution at or Below $300")+
  theme(plot.title = element_text(hjust = 0.5))
  
# The plot reveals a significant portion of the prices are below $100/night.

# Geographical Scatterplot of Prices in Seattle:

# Let's use the ggmap package to load a map of Seattle and visualize which areas
# are more expensive than others.

# Create the map using stamenmap:
seattle_map <- get_stamenmap(bbox = c(left = -122.5, bottom = 47.49, 
                                      right = -122.09, top = 47.74), 
                                      zoom = 9, maptype = "toner")
# Note that the quantiles and price range will influence the pricing scale color:

summary(airbnb$price)
# According to the above summary, we know that Q3 of the IQR was $125, so let's
# determine what percentage of prices are less than or equal to $300.

quantile(airbnb$price)
sum(airbnb$price <=300)/length(airbnb$price)

# ~ 96.5% of listings are <= $300, therefore we filter our price to remove outliers.


# Visualize a "Heatmap" of Seattle with all listing prices included:
ggmap(seattle_map, extent= "normal")+
  geom_point(data = airbnb, 
             aes(x = longitude, y = latitude, color = price), 
             size = 1.5, alpha = .6)+
  scale_color_gradientn(colors = c("mediumblue", "lawngreen", "blueviolet", "red"),
             values = scales::rescale(c(.003, .013, .0176, .025, .2, .3, .4)))+
  xlab("Longitude")+
  ylab("Latitude")+
  ggtitle("Seattle Location Pricing")+
  theme(plot.title = element_text(hjust = 0.5),  
        panel.border = element_rect(color = "gray", fill=NA, size=3))

# It does not appear that one area is significantly more expensive than another
# and that most prices are less than $1000, though outliers may be skewing
# our data, so let's attempt to refine our heatmap by filtering the data.

# Filter the dataframe airbnb_map:
airbnb_map <-airbnb %>% filter(price <=300)

# Visualize a "Heatmap" of Seattle with all listing prices included:
ggmap(seattle_map, extent= "normal")+
  geom_point(data = airbnb_map, 
             aes(x = longitude, y = latitude, color = price), 
             size = 1.5, alpha = .6)+
  scale_color_gradientn(colours = rainbow(5))+
  xlab("Longitude")+
  ylab("Latitude")+
  ggtitle("Seattle Pricing at or Below $300")+
  theme(plot.title = element_text(hjust = 0.5),  
        panel.border = element_rect(color = "gray", fill=NA, size=3))

# The visualization suggests the vast majority of listings in Seattle are 
# between $50 - $150 / night and are relatively concentrated in Seattle 
# proper (as opposed to Newcastle, Mercer Island, or Bellevue).

# Confirm the percentage of listings between $50 - $150 per night:
sum(airbnb$price >= 50 & airbnb$price <= 150)/length(airbnb$price)
# Nearly 70% of listings range from $50 - $150/night.

# Treemap:

# Arrange the cities by listing_sum in a dataframe for the Treemap:
city_distribution <-airbnb %>% group_by(city) %>% summarize(listing_sum = n()) %>%
  arrange(-listing_sum)

# Add column "tmlab" for Treemap labels
city_distribution <-city_distribution %>% 
  unite("tmlab", city:listing_sum, sep = " ", remove = FALSE)

# Plot a Treemap to visualize the distribution of listings by city:
city_distribution %>% ggplot(aes(area = listing_sum, fill = city, label = tmlab))+
  geom_treemap()+
  geom_treemap_text(fontface = "italic", col = "white", place = "center",
                    grow = TRUE)
# The Treemap confirms the overwhelming number of listings in Seattle compared
# to the other cities.

# Visualize Price by City
city_price <-airbnb %>% group_by(city) %>% 
  summarize(mean_price = mean(price),
            listing_sum = n()) %>%
  arrange(-mean_price) %>% mutate(mean_price = sprintf("%0.1f", mean_price))

# Coerce the "mean_price" to integer:
city_price$mean_price <-as.integer(city_price$mean_price)

# Plot the Visualization:
city_price %>% ggplot(aes(x = reorder(city, mean_price), y = mean_price, 
                         fill = city, label = mean_price))+
  geom_bar(stat = "identity")+
  coord_flip()+
  xlab("City")+
  ylab("Mean Price")+
  ggtitle("Mean Price per Night by City")+
  geom_text(size = 4, 
            position = position_stack(vjust = 0.5))+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))
# According to this visualization, it would seem location (latitude & longitude) 
# is highly correlated with price. Let's explore why this was not the case in our
# correlogram. Let's replot the visualization with the sum of listings included.

# Visualize Mean Price & Sum of Listings by City:
# Create dataframe "city_comp" with a percentage column:
city_comp <-city_price %>% 
  mutate(percentage = sprintf("%0.3f",(listing_sum/sum(listing_sum)*100)))

# Add the % symbol to the percentage feature:
city_comp$percentage <- paste(city_comp$percentage, "%")

# Combine the mean price & percentage values into one column:
city_comp <-city_comp %>% 
  unite("citylab", mean_price, percentage, sep = ", ", remove = FALSE)

# Plot the visualization with Mean Price & Percentage of Total Listings:
city_comp %>% 
  ggplot(aes(x = reorder(city, mean_price), y = mean_price, 
                          fill = city, label = citylab))+
  geom_bar(stat = "identity")+
  coord_flip()+
  xlab("City")+
  ylab("Mean Price")+
  ggtitle("Mean Price & Percentage of Total Listings by City")+
  geom_text(size = 4, position = position_stack(vjust = 0.5))+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))

# We learn that over 89% of listings are in Seattle with a mean price of 
# $112/night. The top 7 listings with the highest average prices barely equal
# 1% (~0.8%) of the total listings though their average price is $226.10/night.
# The lower percentage of total listings in these locations at the higher 
# price range, with almost 90% of listings in Seattle likely explains the 
#lower correlation between price and location in this dataset.

# Feature: "rating"

# The "rating" feature is based on a numerical rating from 0 to 5 with a
# mean of 4.841. As this is a relatively high mean rating, we will explore its
# relationship mean price by city to determine its price prediction potential.

# Create the dataframe rating_comp to compare mean price and rating:
rating_comp <-airbnb %>% group_by(city) %>% 
 summarize(mean_rating = mean(rating), mean_price = mean(price)) %>%
  select(city, mean_rating, mean_price)

# Set the parameters for the dual-axis plot:
ylim_1 <-c(0,10)
ylim_2 <-c(70, 400)
b <- diff(ylim_1)/diff(ylim_2)
a <- b*(ylim_1[1] - ylim_2[1])

# Plot the Barplot (Rating) with Overlapping Line (Price):
ggplot(rating_comp, aes(city, group =1))+
  geom_bar(aes(y=mean_rating), stat="identity", color = "navyblue", alpha=.7)+
  geom_line(aes(y = a + mean_price*b), color = "red", size = 2)+
  scale_y_continuous(name = "Mean Rating", 
                     sec.axis = sec_axis(~ (. - a)/b, name = "Mean Price"))+
  xlab("City")+
  ggtitle("Rating & Price Comparison by City")+
  theme(axis.text.x = element_text(angle = 45, hjust=1),
        plot.title = element_text(hjust = 0.5))

range(rating_comp$mean_rating)

# We notice that due to the tight range of mean ratings by city, there is a very 
# low correlation between price and rating. For example, a high price does not
# guarantee a higher rating as the most expensive city Bothell does not have the
# highest rating. Furthermore, cities with lower average prices tend to have
# mean ratings higher than Bothell.

# Feature: "reviews_sum"

# Now let's compare the total number of reviews and different price ranges:
airbnb %>% filter(price <= 1000) %>% 
  ggplot(aes(x = reviews_sum, y = price))+ 
  geom_point(color="navyblue", alpha = 0.6, size = 1.5)+
  xlab("Number of Reviews")+
  ylab("Price")+
  ggtitle("Review vs Price Distribution")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))

# The visualization shows us that there are generally fewer reviews for listings
# with higher prices (above ~$500). We can infer that this is because there are
# fewer stays at these higher-priced listings.


# Feature: "room_type"
# There are 3 different room types: Entire home / apartment, Private Room, &
# Shared Room.

# Let's explore the relationship between room type and average price:
airbnb %>% group_by(room_type) %>%
  summarize(mean_price = mean(price)) %>%
  ggplot(aes(reorder(room_type, mean_price), 
             y = mean_price, label=sprintf("%0.2f", 
                                           round(mean_price, digits = 2))))+ 
  geom_bar(stat = "identity", color = "navyblue", 
           size = 1.5, fill = "deepskyblue")+
  coord_flip()+
  xlab("Room Type")+
  ylab("Mean Price")+
  ggtitle("Mean Price by Room Type")+
  geom_text(size = 3,
            position = position_stack(vjust = 0.5))+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))

# Feature: "bathrooms"

# Bathrooms range in quantity from 0 to 8 in increments of 0.5 per listing.

# Let's explore a visualization that delineates the mean price per listing based
# on the number of bathrooms available:
airbnb %>% group_by(bathrooms) %>% summarize(mean_price = mean(price)) %>%
  ggplot(aes(bathrooms, mean_price))+
  geom_bar(stat = "identity", fill = "deepskyblue", color = "navyblue", size = 1.2)+
  xlab("Number of Bathrooms")+
  ylab("Mean Price per Night")+
  ggtitle("Mean Price per Number of Bathrooms")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))

# We notice that generally, the higher the number of bathrooms per listing, the
# higher the price. Interestingly, listings with 0 bathrooms actually had a
# higher average price than listings with 0.5 - 1.5, and 8 bathrooms! It is
# clear that a listings with 8 bathrooms having a lower mean price than those
# with one bathroom is suspect and most certainly an outlier.

# Now let's visualize a barplot with the total number of listings
# per bathroom count:
airbnb %>% group_by(bathrooms) %>% summarize(sum_bath = length(bathrooms)) %>%
  ggplot(aes(reorder(bathrooms, sum_bath), y=sum_bath, label = sum_bath))+
  geom_bar(stat = "identity", fill = "deepskyblue", color = "cyan", size = 1.2)+
  coord_flip()+
  geom_text(size = 5, color = "navyblue",
            position = position_stack(vjust = 0.5))+
  xlab("Number of Bathrooms")+
  ylab("Total Number of Listings")+
  ggtitle("Listing Distribution by Bathroom Total")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))

# It is evident that the vast majority of listings (~74%) have only 1 bathroom.


# Feature: "bedrooms"

# The bedrooms feature details the number of bedrooms available per listing and
# has been shown in our correlogram to be positively correlated with the price
# per listing.

# Let's explore a visualization that details the mean price per listing versus
# the number of beds available:
airbnb %>% group_by(bedrooms) %>% 
  summarize(mean_price = mean(price)) %>%
  ggplot(aes(bedrooms, mean_price))+
  geom_bar(stat = "identity", 
           color = "navyblue", fill = "deepskyblue", size = 1.5)+
  xlab("Number of Bedrooms")+
  ylab("Mean Price per Night")+
  ggtitle("Mean Price Distribution by Bedroom Total")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))
# The visualization confirms the moderately positive correlation between price 
# and number of bedrooms. This feature will likely contribute to our price
# prediction models.

# Now let's explore listing quantity by number of bedrooms:
airbnb %>% group_by(bedrooms) %>% summarize(sum_beds = length(bedrooms)) %>%
  ggplot(aes(reorder(bedrooms, sum_beds), y = sum_beds, label = sum_beds))+
  geom_bar(stat = "identity", 
           color = "cyan", fill = "deepskyblue", size = 1.5)+
  coord_flip()+
  geom_text(size = 5, color = "navyblue",
            position = position_stack(vjust = 0.5))+
  xlab("Total Number of Listings")+
  ylab("Listing Distribution by Bedroom Total")+
  ggtitle("Number of Bedrooms")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))

# 1 bedroom listings lead the distribution with 4,282 listings (56.5%). As a
# result of the above findings, there is a high probability of a guest renting 
# a 1 bedroom, 1 bath unit.

# Feature: accommodates

# Generally, we might predict that the mean price of a listing per night would 
# increase if it could provide space for more guests.

# Let's explore this idea with a visualization:
airbnb %>% group_by(accommodates) %>% summarize(mean_price = mean(price)) %>%
  ggplot(aes(accommodates, mean_price))+ 
  geom_bar(stat = "identity", color = "navyblue", 
           size = 2, fill = "deepskyblue")+
  xlab("Accommodates")+
  ylab("Mean Price")+
  ggtitle("Number of Guests Accommodated & Mean Price")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))

# The visualization confirms there is a general increase in average price as
# the number of guests able to be accommodated increases.

# Finally, let's check the number of people most units can accommodate:
airbnb %>% group_by(accommodates) %>% 
  summarize(sum_acc = length(accommodates)) %>% 
  ggplot(aes(x = factor(accommodates), y = sum_acc))+
  geom_bar(stat = "identity", color = "navyblue", fill = "deepskyblue", size = 1.5)+
  xlab("Number of Guests Accommodated")+
  ylab("Total Number of Listings")+
  ggtitle("Listings Sum by Accommodation Total")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5))
# The majority of listings can accommodate 4 or fewer guests with 2 being the
# most popular.

# EDA Conclusion:

# The correlogram starts us off by highlighting important relationships between
# price, bedrooms, bathrooms, and accommodates which are later confirmed by 
# visualizations and quantitative analysis. Additionally, we learn location 
# (using latitude and longitude), rating, and reviews sum all have certain 
# potential to increase the accuracy of our predictive models if included in 
# our formula. "Room_type", an important feature, could be vectorized into a 
# quantitative scale, though for the purposes of the following regression models, 
# will be revisited in our conclusion.

### Modeling ###

# Partition Airbnb Combined & Test Sets for the Final Model:
# airbnb_combined will be used to fit the Final Model as it is equal to the 
# sum of the training and validation sets and equal to 90% of the total data.
# airbnb_test, comprised of the remaining 10% of the data, will be used as 
# the test set for the final model.

set.seed(123, sample.kind = "Rounding")
test_index <- createDataPartition(y = airbnb$price, times = 1, p = 0.1, list = F)
airbnb_combined <- airbnb[-test_index,]
airbnb_test <- airbnb[test_index,]

# Remove test_index:
rm(test_index)

# Split Training and Validation Sets from airbnb_combined to train our models:
# airbnb_train will constitute 80% and validation the remaining 20% of
# airbnb_combined.
set.seed(123, sample.kind = "Rounding")
test_index <- createDataPartition(y = airbnb_combined$price, times = 1, 
                                  p = 0.2, list = F)
airbnb_train <- airbnb_combined[-test_index,]
validation <- airbnb_combined[test_index,]

# Remove test_index once again:
rm(test_index)

## The Loss Function / RMSE is defined as follows ##

# Results will be based on calculating the RMSE on the test set.
# The Root Mean Square Error or RMSE weighs large errors more heavily and tends
# to be the preferred measurement of error in regression models when large
# errors are undesirable.

RMSE <-function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Baseline Model: Median Model

# This model will use the median price of the training set to calculate a
# "baseline" RMSE as a benchmark.

airbnb_train_median <-median(airbnb_train$price)

# Table the Results
MM_RMSE <-RMSE(validation$price, airbnb_train_median)
results_table <-tibble(Model_Type = "Baseline Median", RMSE = MM_RMSE) %>% 
  mutate(RMSE = sprintf("%0.2f", RMSE))
knitr::kable(results_table)
# The Baseline Model achieves an RMSE of 99.05.

# Formula: 
# Vectorize the optimal formula that will be used for most Models:
# The formula has been determined by the above EDA as well as experimentation
# on the training models. Additionally, the formula will reduce lines of code.
airbnb_form <-price ~ rating + reviews_sum + bedrooms + bathrooms + 
  accommodates + latitude + longitude

# Linear Model:
lm_airbnb <-lm(airbnb_form, data = airbnb_train)

# Create the prediction
lm_preds <-predict(lm_airbnb, validation)

# Table the Results
LM_RMSE <-RMSE(validation$price, lm_preds)
results_table <-tibble(Model_Type = c("Baseline Median", "Linear"), 
                       RMSE = c(MM_RMSE, LM_RMSE)) %>% 
  mutate(RMSE = sprintf("%0.2f", RMSE))
knitr::kable(results_table)
# The Linear Model vastly improves upon the Baseline Model with an RMSE
# of 73.8.


# Elastic Net Regression with glmnet:
# This model introduces regularization on a generalized linear model using
# penalized maximum likelihood and tuning optimal alpha and lambda parameters.

# Set the seed for reproducibility:
set.seed(123, sample.kind = "Rounding")
train_enr <- train(airbnb_form, data = airbnb_train, method = "glmnet",
                   preProcess = c("center", "scale"),
                   tuneLength = 10, trace = F)
# Confirm the optimal alpha and lambda parameters
train_enr$bestTune
# alpha = 0.1, and lambda = 21.24

# Create the Prediction:
elastic_preds <-predict(train_enr, validation)
# Table the Results
ENR_RMSE <-RMSE(validation$price, elastic_preds)
results_table <-tibble(Model_Type = c("Baseline Median", "Linear",
                                      "Elastic Net Regression"), 
                       RMSE = c(MM_RMSE,LM_RMSE, ENR_RMSE)) %>% 
  mutate(RMSE = sprintf("%0.2f", RMSE))
knitr::kable(results_table)
# The Elastic Net with Regression underperforms the Linear Model with an
# RMSE of 74.06.

# Regression Tree Model with rpart:

# Set the seed for reproducibility:
set.seed(123, sample.kind = "Rounding")
train_rpart <- train(airbnb_form, method = "rpart", data = airbnb_train,
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                     preProcess = c("center", "scale"))

# Check the bestTune to find the final complexity parameter used for the model:
train_rpart$bestTune
# The final cp is 0.00625.

# Create the Prediction:
rt_preds <-predict(train_rpart, validation)

# Table the Results:
RT_RMSE <-RMSE(validation$price, rt_preds)
results_table <-tibble(Model_Type = c("Baseline Median", "Linear",
                                      "Elastic Net Regression", 
                                      "Regression Tree"), 
                       RMSE = c(MM_RMSE,LM_RMSE, ENR_RMSE, RT_RMSE)) %>% 
  mutate(RMSE = sprintf("%0.2f", RMSE))
knitr::kable(results_table)
# The Regression Tree model underperforms the both the Linear Model and
# the Elastic Net Regression Model with an RMSE of 76.03.

# Random Forest Model:

# Set the tuneGrid parameters: 
rf_tune <- expand.grid(.mtry=c(1:3))

# Set the seed for reproducibility:
set.seed(123, sample.kind = "Rounding")
train_rf <- train(airbnb_form, data = airbnb_train,
                  method = "rf", ntree = 150,
                  tuneGrid = rf_tune, nSamp = 1000, 
                  preProcess = c("center","scale"))

# Check the bestTune:
train_rf$bestTune
# The bestTune is a mtry of 1

# Create the Prediction:
rf_preds <-predict(train_rf, validation)

# Table the Results
RF_RMSE <-RMSE(validation$price, rf_preds)
results_table <-tibble(Model_Type = c("Baseline Median", "Linear",
                                      "Elastic Net Regression", 
                                      "Regression Tree", "Random Forest"), 
                       RMSE = c(MM_RMSE,LM_RMSE, ENR_RMSE, RT_RMSE,
                                RF_RMSE)) %>% 
  mutate(RMSE = sprintf("%0.2f", RMSE))
knitr::kable(results_table)
# The Random Forest significantly improves upon the all previous Models with
# an RMSE of 66.57.

# "Bagging Tree" -- Bootstrap Aggregating Model:

# Set the seed for reproducibility:
set.seed(123, sample.kind = "Rounding")
train_bag <-train(airbnb_form, data = airbnb_train, method = "treebag",
                  importance = T, tuneLength = 10, 
                  preProcess = c("center","scale"))

# Create the Prediction
bag_preds <-predict(train_bag, validation)

# Table the Results
BAG_RMSE <-RMSE(validation$price, bag_preds)
results_table <-tibble(Model_Type = c("Baseline Median", "Linear",
                                      "Elastic Net Regression", 
                                      "Regression Tree", "Random Forest",
                                      "BAG"), 
                       RMSE = c(MM_RMSE,LM_RMSE, ENR_RMSE, RT_RMSE,
                                RF_RMSE, BAG_RMSE)) %>% 
  mutate(RMSE = sprintf("%0.2f", RMSE))
knitr::kable(results_table)
# The BAG Tree Model underperforms the Random Forest Model with an RMSE of 68.98

# kNN Model:

# Set the seed for reproducibility:
set.seed(123, sample.kind = "Rounding")
train_knn <- train(airbnb_form, method = "knn", data = airbnb_train,
                   tuneLength = 5, preProcess = c("center","scale"))

# Find best value for k:
train_knn$bestTune
# k = 13 is the final value used for the model.

# Create the Prediction
knn_preds <- predict(train_knn, validation)

# Table the Results
kNN_RMSE <-RMSE(validation$price, knn_preds)
results_table <-tibble(Model_Type = c("Baseline Median", "Linear",
                                      "Elastic Net Regression", 
                                      "Regression Tree", "Random Forest",
                                      "BAG", "kNN"), 
                       RMSE = c(MM_RMSE,LM_RMSE, ENR_RMSE, RT_RMSE,
                                RF_RMSE, BAG_RMSE, kNN_RMSE)) %>% 
  mutate(RMSE = sprintf("%0.2f", RMSE))
knitr::kable(results_table)
# The kNN Model underperforms both the RF and BAG CV Models with an 
# RMSE of 69.94

# Neural Net Model:

# Create the tuneGrid parameters:
NN_grid <-expand.grid(size=c(1, 5, 20), decay = c(0, 0.01, 0.1))

# Set the seed for reproducibility:
set.seed(123, sample.kind = "Rounding")
train_NN <-train(airbnb_form, data = airbnb_train, method= "nnet", 
                 linout = T, trace = F, tuneGrid = NN_grid,
                 preProc = c("center", "scale"))

# Check bestTune:
train_NN$bestTune
# The optimal size is 5 and decay = 0.01

# Create the Prediction:
NN_preds <-predict(train_NN, validation)

# Table the Results:
NN_RMSE <-RMSE(validation$price, NN_preds)
results_table <-tibble(Model_Type = c("Baseline Median", "Linear",
                                      "Elastic Net Regression", 
                                      "Regression Tree", "Random Forest",
                                      "BAG", "kNN", "Neural Net"), 
                       RMSE = c(MM_RMSE,LM_RMSE, ENR_RMSE, RT_RMSE,
                                RF_RMSE, BAG_RMSE, kNN_RMSE, NN_RMSE)) %>% 
  mutate(RMSE = sprintf("%0.2f", RMSE))
knitr::kable(results_table)
# The Neural Net Model underperforms all models except for the Baseline
# Linear & Regression Tree Models with an RMSE of 71.41.

### Final Model ###
# Random Forest Model:
# This model will be fit with airbnb_combined and run on the test set.

# Set the mtry to 1 as determined from the previous tuning on airbnb_train:
tune_grid_rf <-expand.grid(mtry = 1)

# Set the seed for reproducibility:
set.seed(123, sample.kind = "Rounding")
train_rf_final <- train(airbnb_form, data = airbnb_combined,
                        method = "rf", ntree = 150,
                        tuneGrid = tune_grid_rf, nSamp = 1000, 
                        preProcess = c("center","scale"))

# Create the Prediction:
rf_preds_final <-predict(train_rf_final, airbnb_test)

# Table the Results
RFF_RMSE <-RMSE(airbnb_test$price, rf_preds_final)
results_table <-tibble(Model_Type = c("Baseline Median", "Linear",
                                      "Elastic Net Regression", 
                                      "Regression Tree", "Random Forest",
                                      "BAG", "kNN", "Neural Net",
                                      "Random Forest Final"), 
                       RMSE = c(MM_RMSE,LM_RMSE, ENR_RMSE, RT_RMSE,
                                RF_RMSE, BAG_RMSE, kNN_RMSE, NN_RMSE,
                                RFF_RMSE)) %>% 
  mutate(RMSE = sprintf("%0.2f", RMSE))
knitr::kable(results_table)
# As expected, the Random Forest Final Model vastly improved upon the Baseline 
# Model with a final RMSE of 71.43 on the test set -- a 27.88% increase in 
# predictive performance. The Final Model underperformed the RMSE it achieved on 
# the validation set (66.27) as anticipated, though it still outperformed most 
# models except for the BAG, kNN, and Neural Net Models and their RMSEs based on 
# the validation set. (All Models were tested on the test set though only the 
# best performing Model (Random Forest) was selected for the final report).

# In general, ensemble machine learning algorithms that aim to decrease variance 
# were most successful (Bagging Tree and RF). Parallel ensemble methods 
# like Random Forests have proven to be robust price prediction algorithms and 
# therefore it comes as no surprise it was the top performer.
