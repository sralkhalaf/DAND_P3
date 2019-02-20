#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset (Replace this with something more specific!)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row. (Udacity)
# 
# > I think the best field to practice Data Analysis is Health Field, and for that, I choose "No-Show Appointments" dataset   <br/> And I'm aiming at the end of this project to specify some features that can help to determine whether the next patient will show up for his next appointment or not. My questions are as next:
# - Which gender is more likely to miss the appointment?
# - Are patinets with Hypertension, Diabetes, Alcoholism, or Handicap likely to come?
# - Which day of the week patients more likely to not show up for their appointments?

# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties
# 

# In[31]:


# load the dataset using read_csv from pandas, and show head of the dataset
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# > Data Dictionary for this Dataset is powered by (Kaggel):
# - PatientId - Identification of a patient.
# - AppointmentID - Identification of each appointment.
# - Gender = Male or Female.
# - DataMarcacaoConsulta = The day of the actual appointment when they have to visit the doctor.
# - DataAgendamento = The day someone called or registered the appointment, this is before appointment of course.
# - Age = How old is the patient.
# - Neighbourhood = Where the appointment takes place.
# - Scholarship = Ture of False. Is the patient within the Brazilian wealth care system (Bolsa Família) or not.
# - Hipertension = True or False.
# - Diabetes = True or False.
# - Alcoholism = True or False.
# - Handcap = True or False.
# - SMS_received = 1 or more messages sent to the patient.
# - No-show = "Yes" represents the absence of the patient. "No" represent the attendance.
# 
# > In the header, they used the Upper and Lower case to write the name of the columns, except that there is some error in that:
# - Column *"PatinetId"* is written with differnet style than *"AppointmentID"*. The second is more prefare to read and used.
# - Column *"No-show"* use dash instead of Upper and Lower Case.
# - There are missspiling in columns *"Hipertension"* and *"Handcap"*.
# 

# In[32]:


# to show informations regarding the dataset
df.info()


# > From the data above, we can tell that:
# - There is **NOT** any missing value in this dataset.
# - There is **14** columns and **110527** rows\entries.
# - Column *"PatinetId"* have data type (float) while column *"AppointmentID"* have (int).
# - Columns *"ScheduledDay"* and *"AppointmentDay"* stored their value in (object) instead of (date).

# In[33]:


# to check for duplicated 
sum(df.duplicated())


# As we can see, the dataset has no duplicated rows.

# ### Data Cleaning 
# #### Rename Columns
# > To start our Data Cleaning journy we need to rename the columns listed below:
# - Column *"PatinetId"* is written with differnet style than *"AppointmentID"*. The second is more prefare to read and used.
# - Columns *"No-show"* and *"SMS_received"* use dash\underscore instead of Upper and Lower Case.
# - There are missspiling in columns *"Hipertension"* and *"Handcap"*.

# In[22]:


# rename PatientId, Hipertension, Handcap, No-show, SMS_received
df.rename(columns={'PatientId':'PatinetID', 'Hipertension':'Hypertension', 'Handcap':'Handicap',
                   'No-show':'NoShow', 'SMS_received':'SMSReceived'}, inplace=True)


# In[23]:


df.head(1) # to check the new columns.


# #### Fix Columns Data Type
# > To start our Data Cleaning journy we need to rename the columns listed below:
# - Column "PatinetId" have data type (float) while column "AppointmentID" have (int).
# - Columns "ScheduledDay" and "AppointmentDay" stored their value in (object) instead of (date).

# In[24]:


# change PatientID data type from float to int.
df['PatinetID'] = df.PatinetID.astype(int)


# In the above cell, PatientID data type change from float to int

# In[25]:


# change data type for ScheduledDay and AppointmentDay from object to datetime64
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).astype('datetime64[ns]')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).astype('datetime64[ns]')


# In the above cell, ScheduledDay and AppointmentDay has been changed from object to datetime64.

# In[26]:


df.info() # show information about dataframe


# To check for the changed columns.

# #### Add Columns to Dataset
# > To answer one of the above questions, two columns need to be added to the dataset as:
# - *ScheduledDayOfWeek* columns to specify which day of the week the patient schedule an appointmnet.
# - *AppointmentDayOfWeek* columns to specify which day of the week the appointment is.

# In[27]:


# to add new columns (day of week)
df['ScheduledDayOfWeek'] = df.ScheduledDay.dt.weekday_name
df['AppointmentDayOfWeek'] = df.ScheduledDay.dt.weekday_name
df.head(1)


# #### Data Checking
# > Using Unique() function to confirm that all data in this dataset make sence.

# In[28]:


print("Gender: ", np.sort(df.Gender.unique())) # sorting the unique data of gender to confirm if it's ok or not


# Gender in this dataset set either Female or Male.

# In[29]:


print("Age: ", np.sort(df.Age.unique())) # sorting the unique data of age to confirm if it's ok or not


# Age column have (-1) which is not a valid value, and to make sure we gonna run the below statment to make sure that it's a human error in data entery level:

# In[30]:


df.query('Age == "-1"') # to query rows with age = -1


# And to fix that isuue we gonna drop the row.

# In[31]:


df.drop(index=99832, inplace=True) # drop rows with index = 99842 and age = -1


# In[32]:


df.query('Age == "-1"') # query again for age = -1


# After double checking it's seems that everything OK and we can continue our work.

# In[45]:


print("Neighbourhood : ", np.sort(df.Neighbourhood.unique())) # sorting the unique data of neighbourhood to confirm if it's ok or not


# In[46]:


print("Neighbourhood Count: ", np.count_nonzero(df.Neighbourhood.unique())) # count of neighbourhood


# In[47]:


print("Scholarship : ", np.sort(df.Scholarship.unique())) # sorting the unique data of scholarship to confirm if it's ok or not


# In[33]:


# sorting the unique data of conditions to confirm if it's ok or not
print("Hypertension : ", np.sort(df.Hypertension.unique()))
print("\nDiabetes : ", np.sort(df.Diabetes.unique()))
print("\nAlcoholism : ", np.sort(df.Alcoholism.unique()))
print("\nHandicap : ", np.sort(df.Handicap.unique()))


# In[34]:


print("SMSReceived : ", np.sort(df.SMSReceived.unique())) # sorting the unique data of sms received to confirm if it's ok or not


# In[35]:


print("NoShow : ", np.sort(df.NoShow.unique())) # sorting the unique data of no show to confirm if it's ok or not


# So far so good, every columns seems to have the right entries.

# In[36]:


# sorting the unique data of day of week to confirm if it's ok or not
print("ScheduledDayOfWeek : ", np.sort(df.ScheduledDayOfWeek.unique()))
print("\nAppointmentDayOfWeek : ", np.sort(df.AppointmentDayOfWeek.unique()))


# These two columns are the last two in this dataset and they give the right entries.

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# #### Which gender is more likely to miss the appointment?

# To answer this question, we need to split the dataset into two seperate parts and count for each gender.

# In[37]:


# create dataframe with gender = female and no show = missing
df_Female = df.query('Gender == "F" and NoShow == "Yes"')
Gender_F = df_Female['Gender'].value_counts() # count the values of gender
Gender_F


# In[38]:


# create dataframe with gender = male and no show = missing
df_Male = df.query('Gender == "M" and NoShow == "Yes"')
Gender_M = df_Male['Gender'].value_counts()
Gender_M


# In[78]:


plt.pie([Gender_F, Gender_M], labels=["Female", "Male"]);
plt.title("Number of Female or Male That Miss Their Appointments");


# From the chart above, we can answer the question easily. Female patients are more likely to miss their appointments

# #### Are patinets with Hypertension, Diabetes, Alcoholism, or Handicap likely to come?

# In[40]:


# create dataframe with conditions = 1 and they miss the appointmnet 
ConditionYES = df.query('(Hypertension == 1 or Diabetes == 1 or Alcoholism == 1 or Handicap == 1) and NoShow == "Yes"')
ConditionYESCount = ConditionYES.groupby('Gender').count()['Age'] # group by gender
ConditionYESCount


# In[41]:


# create dataframe with conditions = 1 and they attend the appointmnet 
ConditionNO = df.query('(Hypertension == 1 or Diabetes == 1 or Alcoholism == 1 or Handicap == 1) and NoShow == "No"')
ConditionNOCount = ConditionNO.groupby('Gender').count()['Age']
ConditionNOCount


# In[42]:


index = np.array([1,2])
width = 0.3
place = ['Female', 'Male']
plt.bar(index, ConditionNOCount, width=width, label='Show Up')
plt.bar(index + width, ConditionYESCount, width=width, label="Doesn't Show Up")

place = index + (width / 2)
plt.xticks(place, ['Female', 'Male'])
plt.title("Patient With Conditions Show Up By Gender")
plt.xlabel("Gender")
plt.ylabel("Counting of Show Up \ Not Show Up")

plt.legend(bbox_to_anchor=(1,1));


# The graph shows that Female with condtions show up to their appointmnets more than Male with conditions.

# #### Which day of the week patients more likely to not show up for their appointments?

# In[43]:


# create dataframe with day of the week and count of the patinet miss\attend the appointmnet 
dow = df.groupby(['AppointmentDayOfWeek', 'NoShow']).count()['Age']
dowMondayNO = dow['Monday', 'No']
dowMondayYES = dow['Monday', 'Yes']
dowTuesdayNO = dow['Tuesday', 'No']
dowTuesdayYES = dow['Tuesday', 'Yes']
dowWednesdayNO = dow['Wednesday', 'No']
dowWednesdayYES = dow['Wednesday', 'Yes']
dowThursdayNO = dow['Thursday', 'No']
dowThursdayYES = dow['Thursday', 'Yes']
dowFridayNO = dow['Friday', 'No']
dowFridayYES = dow['Friday', 'Yes']
dowSaturdayNO = dow['Saturday', 'No']
dowSaturdayYES = dow['Saturday', 'Yes']


# In[52]:


index = np.array([1,2,3,4,5,6])
width = 0.3
plt.subplots(figsize=(8,6))
place = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
plt.bar(index, [dowMondayNO, dowTuesdayNO, dowWednesdayNO, dowThursdayNO, dowFridayNO, dowSaturdayNO], width=width, label='Show Up')
plt.bar(index + width, [dowMondayYES, dowTuesdayYES, dowWednesdayYES, dowThursdayYES, dowFridayYES, dowSaturdayYES], width=width, label="Doesn't Show Up")

place = index + (width / 2)
plt.xticks(place, ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.title("Counting of Show Up \ Not Show Up Patients By Day")
plt.xlabel("Day")
plt.ylabel("Counting of Show Up \ Not Show Up")
plt.legend(bbox_to_anchor=(1,1));


# In the graph it's clear that pateints likes to go to their appointments during the weekdays not in the weekends

# In[55]:


# creat dataframe with sms received but they miss the appointment
receivedSMS = df.query('SMSReceived == 1 and NoShow == "Yes"')
receivedSMS.Age.plot(kind='hist', x='Age', y='Count How Many Patient Miss The Apointments', title="Patient with Received SMS Missed the Appointment", figsize=(8,8));


# This Chart shows that patients bewteen 20-30 are most likely to miss their appointments even when they received SMS to remind them of the appointment.

# <a id='conclusions'></a>
# ## Conclusions

# 
# > The dataset didn't contain any missing values, neither duplicated values. We change PatientId name and data type, also AppointmentDay and SchedulDay. Also we add two more columns to show which day of the week they schedule their appointments. <br/>
# > One more note, that the dataset doesn't explain more about columns Handicap instead they replace the words with meaningly numbers.<br/>
# >At the end of this project, we can say that there is relationship between the patient likely to show up and these columns:
# - Age
# - Gender
# - Conditions (Hypertension, Diabetes, Alcoholism, Handicap)<br/>
# >And from that analysis we can tell that there is majority of female who shedule and show up for their appointments than male pateints. Also pateints tend to show up for their appointments in the first three days of the week.

# In[ ]:




