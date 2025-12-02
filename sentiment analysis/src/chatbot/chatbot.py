import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SentimentChatbot:
    def __init__(self):
        """
        Automatic Path Finder: Ye code apne aap 'ml' folder dhoond lega.
        """
        print("\nLoading Brain (Model)... 🧠")
        
       
        base_dir = os.path.dirname(os.path.abspath(__file__))

        
        model_path = os.path.join(base_dir, '..', 'ml', 'trained_model.sav')
        vectorizer_path = os.path.join(base_dir, '..', 'ml', 'vectorizer.pkl')

        print(f"Looking for model at: {model_path}") 

        try:
            self.model = pickle.load(open(model_path, 'rb'))
            self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
            print("Model Loaded Successfully! ✅")
        except FileNotFoundError:
            print("\n❌ ERROR: Files nahi mili!")
            print(f"Main dhoond raha tha: {model_path}")
            print("Check karein ki kya 'trained_model.sav' 'src/ml' folder ke andar hai?")
            exit()

        self.stop_words = set(stopwords.words('english'))
        self.port_stem = PorterStemmer()
        self.conversation_history = []

    def preprocess(self, content):
        cleaned_content = re.sub('[^a-zA-Z]', ' ', content)
        cleaned_content = cleaned_content.lower()
        cleaned_content = cleaned_content.split()
        cleaned_content = [self.port_stem.stem(word) for word in cleaned_content if word not in self.stop_words]
        cleaned_content = ' '.join(cleaned_content)
        return cleaned_content

    def predict_sentiment(self, text):
        cleaned_text = self.preprocess(text)
        if not cleaned_text:
            return "Neutral"
        input_vector = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(input_vector)
        if prediction[0] == 4:
            return "Positive"
        else:
            return "Negative"

    def generate_response(self, sentiment):
        if sentiment == "Positive":
            return "That's great! I'm glad to hear that. 😊"
        elif sentiment == "Negative":
            return "Oh no! I'm sorry to hear that. What happened? 😔"
        else:
            return "I see. Please tell me more."

    def start_conversation(self):
        print("\n" + "="*50)
        print("🤖 SENTIMENT BOT IS ONLINE")
        print("Type 'exit' to end.")
        print("="*50 + "\n")

        while True:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'bye', 'quit']:
                break

            sentiment = self.predict_sentiment(user_input)
            bot_reply = self.generate_response(sentiment)
            
            print(f"   -> Detected Sentiment: {sentiment} (Tier 2 Analysis)")
            print(f"Bot: {bot_reply}\n")

            self.conversation_history.append({'user_msg': user_input, 'sentiment': sentiment})

        self.end_conversation_summary()

    def end_conversation_summary(self):
        print("\n" + "="*50)
        print("📊 FINAL CONVERSATION REPORT (Tier 1)")
        print("="*50)
        
        if not self.conversation_history:
            print("No conversation recorded.")
            return

        pos_count = sum(1 for msg in self.conversation_history if msg['sentiment'] == 'Positive')
        neg_count = sum(1 for msg in self.conversation_history if msg['sentiment'] == 'Negative')

        print(f"Total Messages: {len(self.conversation_history)}")
        print(f"Positive: {pos_count} | Negative: {neg_count}")
        print("-" * 30)
        
        if pos_count > neg_count:
            print("🌟 Overall Mood: POSITIVE")
        elif neg_count > pos_count:
            print("🌧️ Overall Mood: NEGATIVE")
        else:
            print("😐 Overall Mood: MIXED/NEUTRAL")
        print("="*50)

if __name__ == "__main__":
    bot = SentimentChatbot()
    bot.start_conversation()