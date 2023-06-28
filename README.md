# Credit-Card-Approval-Prediction-Streamlit-App


[My Streamlit app can be found here!](<https://aswinram1997-credit-card-approval-prediction-streaml-app-mayfo2.streamlit.app/>) 

![Credit Card Approval Prediction App](https://github.com/aswinram1997/Credit-Card-Approval-Prediction-Streamlit-App/assets/102771069/9c644d35-b31f-4899-b3dd-d806c1d0c17f)
  
## Project Overview:
This project aimed to build a [machine learning app](<https://aswinram1997-credit-card-approval-prediction-streaml-app-mayfo2.streamlit.app/>)  to predict whether a credit card application will be approved or rejected based on personal information provided by the applicant and information from credit records. The [Kaggle Dataset](<https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction>) used in this project was pre-processed using various techniques such as binary encoding, one hot encoding, and feature scaling to ensure optimal performance of the models. Two classifiers, logistic regression, and random forest were evaluated based on precision, recall, and F1 score.

The winning model was integrated into a user-friendly Streamlit application, allowing users to interact with the prediction system. The application was deployed using both Streamlit Sharing and Azure cloud. While the Streamlit sharing app is currently available for viewing purposes, proof of deployment on Azure cloud is provided in the Azure_Screenshots folder, showcasing the successful deployment. Overall, this project aimed to leverage machine learning techniques to create an efficient credit card approval prediction system, which was achieved through rigorous data preprocessing, model evaluation, and model deployment.

## Dataset Overview
The credit card approval prediction dataset contains two CSV files: application_record.csv and credit_record.csv:

### application_record.csv:
#### Features:
- ID: Client number
- CODE_GENDER: Gender
- FLAG_OWN_CAR: Is there a car
- FLAG_OWN_REALTY: Is there a property
- CNT_CHILDREN: Number of children
- AMT_INCOME_TOTAL: Annual income
- NAME_INCOME_TYPE: Income category
- NAME_EDUCATION_TYPE: Education level
- NAME_FAMILY_STATUS: Marital status
- NAME_HOUSING_TYPE: Way of living
- DAYS_BIRTH: Birthday
- DAYS_EMPLOYED: Start date of employment
- FLAG_MOBIL: Is there a mobile phone
- FLAG_WORK_PHONE: Is there a work phone
- FLAG_PHONE: Is there a phone
- FLAG_EMAIL: Is there an email
- OCCUPATION_TYPE: Occupation
- CNT_FAM_MEMBERS: Family size

#### Remarks:
- For DAYS_BIRTH and DAYS_EMPLOYED, the values are counted backwards from the current day (0). A value of -1 means yesterday.
- For DAYS_EMPLOYED, if the value is positive, it means the person is currently unemployed.

### credit_record.csv:
#### Features:

- ID: Client number
- MONTHS_BALANCE: Record month (0 is the current month, -1 is the previous month, and so on)
- STATUS: Status of credit record (0: 1-29 days past due, 1: 30-59 days past due, 2: 60-89 days overdue, 3: 90-119 days overdue, 4: 120-149 days overdue, 5: Overdue or bad debts, write-offs for more than 150 days, C: paid off that month, X: No loan for the month)

#### Remarks:
- The STATUS feature indicates the number of days past due or if the credit was paid off for each month.

## Methodology:
The methodology included data exploration, data pre-processing, model training, and evaluation. The dataset was split into training, validation, and testing sets to prevent overfitting. Data pre-processing techniques were employed to optimize the model's performance. The random search CV with 3 fold cross-validation was used to identify the best algorithm and its corresponding hyperparameters. Precision, recall, and F1 score were used to evaluate model performance.

## Results:
The random forest classifier model outperformed the logistic regression model and was identified as the winning model. The model's performance was evaluated using precision, recall, and F1 score, which were found to be satisfactory. The model did not overfit on the training set and generalized well on the test set. This model is used for developing the Streamlit app.

## Conclusion:
This project demonstrated the potential of machine learning in automating the credit card approval process. The results obtained from this project highlight the need for further exploration of data pre-processing techniques, hyperparameter tuning, and other algorithms to improve the model's performance. The findings of this project suggest that the use of machine learning can provide banks with a valuable tool to objectively quantify the credit risk associated with a particular applicant and provide a standard measure for credit risk assessment. Lastly, the winning model was deployed using both Streamlit sharing and Azure. Deployment showcased the practical application of the project's findings and the potential for standardized credit card approval in the banking industry.
