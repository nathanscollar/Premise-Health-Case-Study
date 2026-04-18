This Github Repo contains my case study for Premise Health, focused on the data science and machine learning analysis of a claims based dataset. There are two main spreadsheets,
the first is "Medical Claims Data" and contains all of the claims submitted by a patient, including the place of service, cost/paid amount, diagnosis, and procedure. The second
spreadsheet is "member_data_with_surveys" and contains the patient's age, gender, whether or not they have a variety of health-related conditions, and their feedback. Using this
data, the first thing I did was sentiment analysis, where I essentially set up a reference list of positive and negative words, encoded them, and calculated the centroid. From there,
I encode the patient's feedback and calculate cosine similarity to both the positive words and the negative words, and take the difference between the two to get the sentiment score.
A positive sentiment score (> 0) represents positive feedback, whereas a negative sentiment score (< 0) represents negative feedback. I then used this binary sentiment variable for
logistic regression, where I found that within a 95% confidence interval, individuals with asthma, chronic pain, depression, hyperlipidemia, hypertension, and lower back pain are more 
likely to give negative feedback. After this, I looked at the correlation between amount of money paid for treatment and sentiment score. I ran a logistic regression with the feature
being total amount paid and the output being sentiment score, and got a t-statistic of -4.136 with a p-value under 0.05. The slope of the logistic regression was 3.78e-07. These values
indicate that as amount paid increases, the sentiment score decreases. I also binned the amount paid into different groups to further analyze the results, and realized that patients
in the lowest bin ($0-1,000 spent) had the highest rate of positive feedback (61.04%), whereas patients in the highest bit ($50,000+ spent) had the lowest rate of positive feedback
(31.3%). I began making setting up a patient search RAG pipeline, but did not have time to complete it. The idea was that someone could enter in some query about a patient or the
dataset as a whole, the query would be classified as factual or semantic, and be routed differently depending on that classification. Factual queries could be answered by a large
language model writing Python Pandas code (such as, "does member_id 7 have ADHD?"), whereas semantic queries would be answered through similarity search (such as "how was member_id 
9's overall experience), where the full response would be written by a large language model.