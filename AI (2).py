import os
import pickle
import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer 

# Load the trained model
nlp_model = pickle.load(open("C:/Users/Ramu M/NLP_model.pkl", "rb"))

# Sidebar setup
with st.sidebar:
    image = Image.open(r"C:\Users\Ramu M\Downloads\AI chatbot image.jpg")
    st.image(image, caption='🤖 Smart Conversational Partner', width=250)

    selected = option_menu('AI Echo: Your Smartest Conversational Partner',
        
                           ['💬 AI Echo Sentiment Analysis',
                            '📈Sentiment Analysis']

    )

if selected == "💬 AI Echo Sentiment Analysis":
    st.title(":red[AI Echo: Your Smartest Conversational Partner]")
    st.markdown("### Fill out the form below to analyze review sentiment.")

    # Form for user input
    with st.form("📋form"):
        title = st.text_input("📌 Title")
        rating = st.selectbox("⭐ Rating", [1, 2, 3, 4, 5])
        username = st.text_input("👤 Username")
        platform = st.text_input("💻 Platform")
        language = st.text_input("🗣️ Language")
        location = st.text_input("📍 Location")
        version = st.text_input("🔢 Version")
        verified_purchase = st.selectbox("✔️ Verified Purchase", ["Yes", "No"])
        clean_review = st.text_input("📝 Clean Review")

        # ✅ Move the submit button inside the form
        submitted = st.form_submit_button("🎯 Predict Sentiment")

    # This runs only if the form is submitted
    if submitted:
        verified_bool = True if verified_purchase == "Yes" else False

        input_df = pd.DataFrame([{
            'title': title,
            'rating': rating,
            'username': username,
            'platform': platform,
            'language': language,
            'location': location,
            'version': version,
            'verified_purchase': verified_bool,
            'clean_review': clean_review
        }])

        # Encode categorical columns just like training
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except ValueError:
                input_df[col] = input_df[col].astype('category').cat.codes

        prediction = nlp_model.predict(input_df)

        # Display result
        st.markdown("---")
        st.subheader("📈 Sentiment Prediction")
        if prediction[0] == 1:
            st.success("😊 Positive Sentiment Detected")
            st.info("""**Recommended:**
- Keep up the great work!
- Thank you for sharing your positive experience.
- Your review helps others make informed decisions.""")
        elif prediction[0] == 2:
            st.success("😐 Neutral Sentiment Detected")
            st.info("""**Suggestions to improve review:**
- Provide more details.
- Highlight key features you liked or disliked.""")
        else:
            st.error("😠 Negative Sentiment Detected")
            st.info("""**Suggestions for Improvement**
- Clarify the review.
- Offer constructive feedback.""")



if selected == "📈Sentiment Analysis":
    st.title("🧠 1. Overall Sentiment of User Reviews")
    st.markdown("### 📊 Sentiment Distribution - Positive, Neutral, Negative")

    file_path = "C:/Users/Ramu M/sentiment_data_cleaned.csv"

    try:
        df = pd.read_csv(file_path)

        if 'sentiment' not in df.columns:
            st.error("❌ The dataset must contain a 'sentiment' column.")
        else:
            sentiment_counts = df['sentiment'].value_counts()
            sentiment_percent = (sentiment_counts / len(df)) * 100

            st.markdown("### 📃 Sentiment Summary")
            st.write(sentiment_percent.round(2).astype(str) + "%")

            st.markdown("### 🥧 Pie Chart")
            fig1, ax1 = plt.subplots()
            ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                    colors=['#8BC34A', '#FFC107', '#F44336'])
            ax1.axis('equal')
            st.pyplot(fig1)


    except FileNotFoundError:
        st.error("🚫 File not found. Please check the path and try again.")


    st.markdown("## 2. ⭐ Sentiment vs Rating Analysis")
    st.write("Let's explore if user ratings align with the sentiment predicted from review text.")
    file_path = "C:/Users/Ramu M/sentiment_data_cleaned.csv"
# Grouping by rating and sentiment
    rating_sentiment_counts = df.groupby(['rating', 'sentiment']).size().unstack().fillna(0)

# Plotting
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    rating_sentiment_counts.plot(kind='bar', stacked=True, ax=ax3,
                             color=['#F44336', '#FFC107', '#8BC34A'])  # Neg, Neutral, Pos
    ax3.set_title("Sentiment Distribution Across Ratings")
    ax3.set_xlabel("User Rating (Stars)")
    ax3.set_ylabel("Number of Reviews")
    ax3.legend(["Negative", "Neutral", "Positive"])
    st.pyplot(fig3)

# Mismatch detection
    st.markdown("### ⚠️ Mismatch between Rating and Sentiment")

# For instance: 1-star but Positive sentiment
    mismatch_positive = df[(df['rating'] == 1) & (df['sentiment'] == 1)]
    mismatch_negative = df[(df['rating'] == 5) & (df['sentiment'] == 0)]

    st.write(f"- 🔄 **1-star reviews with Positive Sentiment**: {len(mismatch_positive)}")
    st.write(f"- 🔄 **5-star reviews with Negative Sentiment**: {len(mismatch_negative)}")

# Show a few mismatched reviews
    if not mismatch_positive.empty:
       st.markdown("#### 🤔 Examples: 1-Star but Positive Sentiment")
       st.dataframe(mismatch_positive[['rating', 'clean_review']].head())

    if not mismatch_negative.empty:
       st.markdown("#### 🤔 Examples: 5-Star but Negative Sentiment")
       st.dataframe(mismatch_negative[['rating', 'clean_review']].head())



#3



    @st.cache_data
    def load_data():
           df = pd.read_csv("C:/Users/Ramu M/AI_NLP.csv")
           df['clean_review'] = df['clean_review'].astype(str).str.strip()
           df = df[df['clean_review'].notna() & (df['clean_review'] != "")]
           return df

    df = load_data()
    st.title("🔍3. Sentiment-wise Keyword Analysis")

    if 'clean_review' not in df.columns or 'sentiment' not in df.columns:
       st.error("The dataset must contain 'clean_review' and 'sentiment' columns.")
    else:
       sentiment_classes = df['sentiment'].unique()
       selected_sentiment = st.sidebar.selectbox("Select Sentiment Class", sentiment_classes)

    filtered_df = df[df['sentiment'] == selected_sentiment]

    # Combine all words from the reviews
    all_reviews = filtered_df['clean_review'].dropna().astype(str).tolist()
    all_words = []
    for review in all_reviews:
        words = review.strip().split()
        all_words.extend(words)

    if len(all_words) == 0:
        st.warning(f"No valid words found to generate word cloud for sentiment: '{selected_sentiment}'")
    else:
        word_freq = Counter(all_words)
        most_common_words = word_freq.most_common(30)

        # Word Cloud
        st.subheader(f"Word Cloud for Sentiment '{selected_sentiment}'")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Frequency Table
        st.subheader(f"Top Keywords for Sentiment '{selected_sentiment}'")
        freq_df = pd.DataFrame(most_common_words, columns=['Keyword', 'Frequency'])
        st.dataframe(freq_df)


    
#4


    st.subheader("📊 4. Analyze sentiment trends by month or week")
    #4# Load CSV data
    df = pd.read_csv(r"C:/Users/Ramu M/NLP.csv")

# Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # replace 'date' with actual column name if different

# Drop rows with invalid dates
    df = df.dropna(subset=['date'])

# Option to group by month or week
    grouping = st.radio("View sentiment trend by:", ["Month", "Week"])

# Create a new column for grouping
    if grouping == "Month":
       df['time_group'] = df['date'].dt.to_period('M').astype(str)
    else:
       df['time_group'] = df['date'].dt.to_period('W').astype(str)

# Group by time and sentiment
    trend = df.groupby(['time_group', 'sentiment']).size().unstack().fillna(0)

# Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    trend.plot(ax=ax, marker='o')
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Reviews")
    ax.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)




#5 


# Load your dataset
    df = pd.read_csv("C:/Users/Ramu M/AI_NLP.csv")

# Check required columns exist

    st.subheader("📊 5. Sentiment Distribution Comparison (Verified vs Non-Verified)")
    
    # Count plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x='sentiment', hue='verified_purchase', palette='Set2', ax=ax)
    ax.set_title('Sentiment Distribution by Verified Purchase')
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)
    
    # Show counts
    st.subheader("🔢 Raw Sentiment Counts")
    count_data = df.groupby(['verified_purchase', 'sentiment']).size().unstack().fillna(0).astype(int)
    st.dataframe(count_data)

    # Show percentage
    st.subheader("📈 Percentage Breakdown")
    percent_data = count_data.div(count_data.sum(axis=1), axis=0) * 100
    st.dataframe(percent_data.round(2))

    



#6


    # Load your dataset
    df = pd.read_csv("C:/Users/Ramu M/AI_NLP.csv")

# Check necessary columns exist
    if 'clean_review' in df.columns and 'sentiment' in df.columns:

      st.subheader("📝 6. Review Length vs Sentiment")

    # Add review length column
    df['review_length'] = df['clean_review'].astype(str).apply(len)

    # Plot average review length per sentiment
    st.markdown("📊 Average Review Length by Sentiment")
    avg_length = df.groupby('sentiment')['review_length'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=avg_length, x='sentiment', y='review_length', palette='coolwarm', ax=ax)
    ax.set_title("Average Review Length by Sentiment")
    ax.set_ylabel("Average Length (characters)")
    st.pyplot(fig)

    # Optional: Show table
    st.subheader("📄 Table of Average Lengths")
    st.dataframe(avg_length.rename(columns={'review_length': 'Average Length'}).round(2))


#7


    # Load your dataset
    df = pd.read_csv("C:/Users/Ramu M/AI_NLP.csv")

# Check required columns
    if 'location' in df.columns and 'sentiment' in df.columns:

       st.title("📍7. Sentiment by User Location")

    # Drop missing location values
    df = df.dropna(subset=['location'])

    # Count sentiment per location
    location_sentiment = df.groupby(['location', 'sentiment']).size().unstack(fill_value=0)

    # Show top N locations with most reviews (optional)
    top_locations = location_sentiment.sum(axis=1).sort_values(ascending=False).head(10).index
    filtered = location_sentiment.loc[top_locations]

    # Plot
    st.subheader("🌍 Sentiment Distribution in Top Locations")
    fig, ax = plt.subplots(figsize=(10, 5))
    filtered.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)
    ax.set_title("Sentiment Breakdown by Location")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

    # Optional table
    st.subheader("📄 Sentiment Counts by Location")
    st.dataframe(filtered)



 #8


# Load the dataset
    st.subheader("📊 8.Sentiment Distribution Across Platforms (Web vs Mobile)")
    df = pd.read_csv("C:/Users/Ramu M/sentiment_data_cleaned.csv")

# Fix sentiment values: only 'positive', 'neutral', 'negative' are allowed
    valid_sentiments = ['positive', 'neutral', 'negative']
    df['sentiment'] = df['sentiment'].apply(lambda x: x if x in valid_sentiments else 'neutral')

# Map platform: 0 = Web, 1 = Mobile
    platform_mapping = {0: 'Web', 1: 'Mobile'}
    df['platform_name'] = df['platform'].map(platform_mapping)

# Sentiment distribution across platforms
    sentiment_platform = df.groupby(['platform_name', 'sentiment']).size().reset_index(name='count')

# Show dataframe

    st.dataframe(sentiment_platform)

# Plot

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=sentiment_platform, x='platform_name', y='count', hue='sentiment', palette='Set2', ax=ax)
    plt.title("Sentiment Distribution by Platform")
    plt.xlabel("Platform")
    plt.ylabel("Number of Reviews")
    st.pyplot(fig)

# Optional: Insightful text
    web_negative = sentiment_platform[(sentiment_platform['platform_name'] == 'Web') & (sentiment_platform['sentiment'] == 'negative')]['count'].values
    mobile_negative = sentiment_platform[(sentiment_platform['platform_name'] == 'Mobile') & (sentiment_platform['sentiment'] == 'negative')]['count'].values

    if web_negative.any() and mobile_negative.any():
        if web_negative[0] > mobile_negative[0]:
           st.warning("⚠️ More negative reviews on **Web** – consider improving the web user experience.")
        elif mobile_negative[0] > web_negative[0]:
           st.warning("⚠️ More negative reviews on **Mobile** – consider enhancing the mobile experience.")

 
#9



 #Load dataset
    df = pd.read_csv("C:/Users/Ramu M/AI_NLP.csv")
    st.subheader("📄 9. Sentiment Counts per ChatGPT Version")
# Clean sentiment (ensure only valid values)
    valid_sentiments = ['positive', 'neutral', 'negative']
    df['sentiment'] = df['sentiment'].apply(lambda x: x if x in valid_sentiments else 'neutral')

# Drop rows with missing version or sentiment
    df = df.dropna(subset=['version', 'sentiment'])

# Group data
    version_sentiment = df.groupby(['version', 'sentiment']).size().reset_index(name='count')

# Sort versions for better visualization
    version_order = df['version'].value_counts().index.tolist()
    version_sentiment['version'] = pd.Categorical(version_sentiment['version'], categories=version_order, ordered=True)

# Show table
    st.dataframe(version_sentiment)
# Barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=version_sentiment, x='version', y='count', hue='sentiment', palette='Set1', ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("ChatGPT Version")
    plt.ylabel("Number of Reviews")
    plt.title("User Sentiment per Version")
    st.pyplot(fig)

# Optional Insight
    highest_neg = version_sentiment[version_sentiment['sentiment'] == 'negative'].sort_values('count', ascending=False).head(1)
    lowest_neg = version_sentiment[version_sentiment['sentiment'] == 'negative'].sort_values('count', ascending=True).head(1)

    if not highest_neg.empty:
        st.warning(f"⚠️ Version **{highest_neg.iloc[0]['version']}** received the most negative reviews.")
    if not lowest_neg.empty:
        st.success(f"✅ Version **{lowest_neg.iloc[0]['version']}** had the fewest negative reviews.")



#10


    st.subheader("💔 10. Top Negative Feedback Keywords")

# Load data
    df = pd.read_csv("C:/Users/Ramu M/AI_NLP.csv")
    df['clean_review'] = df['clean_review'].astype(str)
    st.write("Unique sentiment labels:", df['sentiment'].unique())

# Filter negative reviews and clean
    neg_reviews = df[df['sentiment'].str.lower() == 'negative']['clean_review']

    neg_reviews = [review.strip() for review in neg_reviews if review.strip()]  # remove empty or whitespace-only
    st.write("Sentiment column unique values:")
    st.write(df['sentiment'].value_counts())

    st.write("Sample data:")
    st.dataframe(df[['sentiment', 'clean_review']].head(10))


# Check if we have valid reviews
    if len(neg_reviews) == 0:
       st.warning("No valid negative reviews found after cleaning.")
    else:
    # Vectorize the reviews
       vectorizer = CountVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(neg_reviews)
        # Get top keywords
        word_counts = X.sum(axis=0).A1
        keywords = vectorizer.get_feature_names_out()
        word_freq = dict(zip(keywords, word_counts))

        # Top 15 keywords
        top_keywords = Counter(word_freq).most_common(15)
        top_words_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=top_words_df, x='Frequency', y='Keyword', palette='Reds_r')
        ax.set_title("Most Common Keywords in Negative Reviews")
        st.pyplot(fig)
    except ValueError:
        st.error("Failed to vectorize reviews: resulting vocabulary was empty. Check the review content.")

    
