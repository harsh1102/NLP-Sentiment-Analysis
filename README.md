# Sentiment Analysis Of Twitter Data for a movie review using LSTM


We have applied Long Short-Term Memory (LSTM) algorithm to a Twitter dataset for the purpose of binary sentiment analysis.
We architected a deep learning neural network model to achieve the accuracy of 94% on static dataset and then used same model on real-time data.
Our dataset consists of 50,000 instances of data which are tweets and all the data is labeled as ‘negative’ or ‘positive’, meaning binary classification.
Analyzed the model using accuracy and loss as the measurements. Also observed the effect on accuracy and loss when real-time data was used as testing data instead of splitting the training data.

## Model Architecture

![image](https://user-images.githubusercontent.com/41270424/235325728-3fdb1f6d-804d-4468-ad1f-7b3a8ae48c00.png)

## Result Plots for LSTM (Model Loss)

![image](https://user-images.githubusercontent.com/41270424/235325843-aeecd1c6-8de8-4188-992d-3b3382b36728.png)

## Result Plots for LSTM (Model Accuracy)

![image](https://user-images.githubusercontent.com/41270424/235325851-21ecbe9f-ab07-4b1a-9fab-e500853d2a74.png)

## What is different from our reference papers

Referenced paper have used static data for testing purposes but we made analysis based on the live feed data and observe the changes in the result as live data can be very different from a preprocessed dataset. 
Reference papers had an accuracy between 70-84 % but we achieved accuracy rates of more than 93% by using proper data filtration and cleansing techniques.
Our model was trained on dataset but was validated on the real world data which is the scenario in real world. Whereas, reference papers used train-test split of data.

### Enhancement to refereed work

The low accuracy of the initial models necessitated the implementation of more    advanced Deep Learning models, such as Recurrent Neural Networks (RNNs). RNNs possess a memory unit that allows for the incorporation of prior input when predicting current output. Long Short-Term Memory (LSTM) network is typically employed to process large, temporal datasets. The models performance was satisfactory on a large Movie tweets dataset. LSTM along with right fit of the complexity of model and the dataset helped us to score higher Accuracy and lower loss on the dataset.

## Future-work

Use and observe other Deep learning model instead of LSTM for the same functionalities to see if better results can be achieved in terms of time and space complexity.
Explore other data manipulation techniques to better fit your data to models and filter data for real instances. This is because there is room for improvement.




