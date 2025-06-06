import smtplib
import ssl
import os
import subprocess
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import imapclient
import email
from email.header import decode_header
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.base import LLM
from langchain_core.prompts import PromptTemplate  
from langchain.chains import LLMChain               
from pydantic import Field
from langchain_community.llms import Ollama  # Only this Ollama import is needed

# Load environment variables from a .env file if available
load_dotenv()

class OllamaLLM(LLM):
    """
    A LangChain-compatible wrapper that calls Ollama via subprocess.
    Fields declared as Pydantic fields so they can be set on instantiation.
    """
    model: str = Field(default="llama3.2:latest")
    executable: str = Field(default="/snap/bin/ollama")

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: list = None) -> str:
        """
        Executes Ollama with the provided prompt.
        The prompt is passed as a positional argument with no '--prompt' flag.
        """
        command = [self.executable, "run", self.model, prompt]
        print("\n[DEBUG] Executing Ollama command in LLM wrapper:")
        #print("[DEBUG] Command:", command)
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("[DEBUG] Ollama output (LLM):", result.stdout)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print("[DEBUG] Error calling Ollama in LLM wrapper:")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return "Unable to generate response."







def send_email(smtp_server: str, port: int, sender_email: str, password: str,
               receiver_email: str, subject: str, email_body: str,
               attachment_path: str = None) -> None:
    """
    Composes and sends an email with the provided subject, body and optional attachment.
    """
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Attach the email body as plain text.
    message.attach(MIMEText(email_body, "plain"))
    print("[DEBUG] Email body attached.")

    if attachment_path:
        print("[DEBUG] Attachment provided:", attachment_path)
        if os.path.isfile(attachment_path):
            try:
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",
                                f"attachment; filename={os.path.basename(attachment_path)}")
                message.attach(part)
                print("[DEBUG] Attachment attached successfully.")
            except Exception as e:
                print(f"[DEBUG] Error attaching file {attachment_path}: {e}")
        else:
            print(f"[DEBUG] Attachment file not found: {attachment_path}")

    print("[DEBUG] Connecting to SMTP server.")
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            print("[DEBUG] Logging in to SMTP server as:", sender_email)
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("\n[DEBUG] Email sent successfully!")
    except Exception as e:
        print("[DEBUG] Failed to send email:", e)

   
def generate_body(subject: str) -> str:
    """
    Generates a summarized email body using Ollama via our custom LangChain LLM.
    We use an LLMChain with a prompt template that instructs the AI
    to produce a succinct summary.
    """
    prompt_template = PromptTemplate(
        input_variables=["subject"],
        # Note the change: requesting a concise summarized version of the email body.
        template="Generate a concise and summarized and crisp ony with 3 lines email body without salutation,do not generate 'Here is a concise and summarized email body' in it based on the following subject: {subject}\n\nEmail body:"
                 "Ensure the response has **specific details** filled in, without placeholders or missing fields. "
                 "do not use names of personalities until specified"
                 "Use realistic data and DO NOT leave blanks like '[amount]' or '[Recipient's Name] or 'x/y/z' or [date]"    
                 "DONOT add your own content use inly the given data"
    )                   
    ollama_llm = OllamaLLM()  # Using default model and executable.
    chain = prompt_template | ollama_llm
    print("\n[DEBUG] Running LLMChain with subject:", subject)
    body = chain.invoke({'subject': subject})
    return body if body.strip() else "Could not generate email body."
 

# ...existing imports...
def classifier():
    def decode_subject(subject):
        """Decodes email subject to readable string."""
        decoded = decode_header(subject)
        subject_str = ''
        for s, enc in decoded:
            if isinstance(s, bytes):
                subject_str += s.decode(enc or 'utf-8', errors='ignore')
            else:
                subject_str += s
        return subject_str

    def extract_email_body(email_msg):
        """Extracts plain text body from email message."""
        if email_msg.is_multipart():
            for part in email_msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode("utf-8", errors="ignore")
        else:
            return email_msg.get_payload(decode=True).decode("utf-8", errors="ignore")
        return "No readable content found."

    def train_spam_classifier():
        # Minimal example data; replace with a larger labeled dataset for real use
        data = [
            {"text": "Congratulations! You won a lottery. Claim now.", "label": 1},  # spam
            {"text": "Meeting at 10am tomorrow.", "label": 0},  # not spam
            {"text": "Get cheap meds online!", "label": 1},  # spam
            {'text': 'attached document','label':0} ,  # not spam 
            {"text": "Please find attached the report.", "label": 0},  # not spam
            {"text": "You have been selected for a prize.", "label": 1},  # spam
            {"text": "Lunch at 1pm?", "label": 0},  # not spam
            {"text": "Don't miss our exclusive offer!", "label": 1},  # spam
            {"text": "Your invoice is ready for download.", "label": 0},  # not spam
            {"text": "Urgent: Update your account information.", "label": 1},  # spam
            {"text": "Reminder: Your appointment is tomorrow.", "label": 0},  # not spam
            {"text": "Limited time offer just for you!", "label": 1},  # spam
            {"text": "New product launch!", "label": 1},# spam
            {"text": "your amazon order has been shipped", "label": 0},  # not spam
            {"text": "Get paid to work from home!", "label": 1},  # spam
            {"text": "Your subscription has been renewed.", "label": 0},  # not spam
            {"text": "Claim your free gift now!", "label": 1},  # spam
            {"text": "Your bank statement is ready.", "label": 0},  # not spam
            {"text": "Congratulations! You have won a free vacation!", "label": 1},  # spam
            {"text": "Your package has been delivered.", "label": 0},  # not spam
            {"text": "Get rich quick with this simple trick!", "label": 1},  # spam
            {"text": "Your account has been compromised, reset your password.", "label": 1},  # spam
            {"text": "Don't forget to submit your timesheet.", "label": 0},  # not spam
            {"text": "Exclusive deal just for you!", "label": 1},  # spam
            {"text": "Your flight details are confirmed.", "label": 0},  # not spam
            {"text": "Act now! Limited time offer!", "label": 1},  # spam
            {"text": "Your subscription will expire soon.", "label": 0},  # not spam
            {"text": "You have a new message from your friend.", "label": 0},  # not spam
            {"text": "Congratulations! You've been selected for a special offer!", "label": 1},  # spam
            {"text": "Your order has been shipped.", "label": 0},  # not spam
            {"text": "Get paid to take surveys online!", "label": 1},  # spam
            {"text": "Your account has been suspended.", "label": 0},  # not spam
            {"text": "Reminder: Your bill is due soon.", "label": 0},  # not spam
            {"text": "You have a new follower on social media.", "label": 0},  # not spam
            {"text": "Congratulations! You've won a gift card!", "label": 1},  # spam
            {"text": "Your subscription has been cancelled.", "label": 0},  # not spam
            {"text": "Get a free trial of our premium service!", "label": 1},  # spam
            {"text": "Your package is ready for pickup.", "label": 0}, # not spam
            {"text": "Don't miss out on this exclusive offer!", "label": 1},  # spam
            {"text": "Your account has been verified.", "label": 0}, # not spam
            {"text": "Congratulations! You've been selected for a free gift!", "label": 1},  # spam
            {"text": "Your payment has been processed successfully.", "label": 0},  # not spam
            {"text": "Get a free consultation with our experts!", "label": 1},  # spam
            {"text": "Your account settings have been updated.", "label": 0},  # not spam
            {"text": "Congratulations! You've won a cash prize!", "label": 1},  # spam
            {"text": "Your subscription has been renewed successfully.", "label": 0},  # not spam
            {"text": "Get a free e-book on how to make money online!", "label": 1},  # spam
            {"text": "Your account has been locked due to suspicious activity.", "label": 0},  #not spam
            {"text": "Reminder: Your appointment is tomorrow at 3 PM.", "label": 0},  # not spam
            {'text': 'your invoice is ready for download', 'label': 0},  # not spam
            {"text": "Please find attached the report.", "label": 0},  # not spam
        
        ]
        df = pd.DataFrame(data)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]
        model = LogisticRegression()
        model.fit(X, y)
        return model, vectorizer

    spam_model, spam_vectorizer = train_spam_classifier() 
    label_map = {0: "Not Spam", 1: "Spam"}

    def predict_spam(email_text):
        X_new = spam_vectorizer.transform([email_text])
        pred = spam_model.predict(X_new)[0]
        return pred  # 1 for spam, 0 for not spam


    def classify_inbox_emails():
        SENDER_EMAIL = os.getenv("SENDER_EMAIL")
        EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
        output = []
        with imapclient.IMAPClient('imap.gmail.com') as client:
            client.login(SENDER_EMAIL, EMAIL_PASSWORD)
            client.select_folder("INBOX")
            messages = client.search("ALL")  # Fetch all emails

            for msg_id in messages:
                raw_message = client.fetch([msg_id], ["RFC822"])[msg_id][b"RFC822"]
                email_msg = email.message_from_bytes(raw_message)
                subject = decode_subject(email_msg["Subject"]) if email_msg["Subject"] else "(No Subject)"
                body = extract_email_body(email_msg)
                spam_pred = predict_spam(body)
                output.append(f"Subject: {subject}")
                output.append(f"Prediction: {label_map[spam_pred]}")
                output.append(f"Body (first 100 chars): {body[:100]}\n{'-'*40}")
        return "\n".join(output)

    # --- Spam Classifier Training ---
    def train_spam_classifier():
        # Minimal example data; replace with a larger labeled dataset for real use
        data = [
            {"text": "Congratulations! You won a lottery. Claim now.", "label": 1},  # spam
            {"text": "Meeting at 10am tomorrow.", "label": 0},  # not spam
            {"text": "Get cheap meds online!", "label": 1},  # spam
            {'text': 'attached document','label':0} ,  # not spam 
            {"text": "Please find attached the report.", "label": 0},  # not spam
            {"text": "You have been selected for a prize.", "label": 1},  # spam
            {"text": "Lunch at 1pm?", "label": 0},  # not spam
            {"text": "Don't miss our exclusive offer!", "label": 1},  # spam
            {"text": "Your invoice is ready for download.", "label": 0},  # not spam
            {"text": "Urgent: Update your account information.", "label": 1},  # spam
            {"text": "Reminder: Your appointment is tomorrow.", "label": 0},  # not spam
            {"text": "Limited time offer just for you!", "label": 1},  # spam
            {"text": "New product launch!", "label": 1},# spam
            {"text": "your amazon order has been shipped", "label": 0},  # not spam
            {"text": "Get paid to work from home!", "label": 1},  # spam
            {"text": "Your subscription has been renewed.", "label": 0},  # not spam
            {"text": "Claim your free gift now!", "label": 1},  # spam
            {"text": "Your bank statement is ready.", "label": 0},  # not spam
            {"text": "Congratulations! You have won a free vacation!", "label": 1},  # spam
            {"text": "Your package has been delivered.", "label": 0},  # not spam
            {"text": "Get rich quick with this simple trick!", "label": 1},  # spam
            {"text": "Your account has been compromised, reset your password.", "label": 1},  # spam
            {"text": "Don't forget to submit your timesheet.", "label": 0},  # not spam
            {"text": "Exclusive deal just for you!", "label": 1},  # spam
            {"text": "Your flight details are confirmed.", "label": 0},  # not spam
            {"text": "Act now! Limited time offer!", "label": 1},  # spam
            {"text": "Your subscription will expire soon.", "label": 0},  # not spam
            {"text": "You have a new message from your friend.", "label": 0},  # not spam
            {"text": "Congratulations! You've been selected for a special offer!", "label": 1},  # spam
            {"text": "Your order has been shipped.", "label": 0},  # not spam
            {"text": "Get paid to take surveys online!", "label": 1},  # spam
            {"text": "Your account has been suspended.", "label": 0},  # not spam
            {"text": "Reminder: Your bill is due soon.", "label": 0},  # not spam
            {"text": "You have a new follower on social media.", "label": 0},  # not spam
            {"text": "Congratulations! You've won a gift card!", "label": 1},  # spam
            {"text": "Your subscription has been cancelled.", "label": 0},  # not spam
            {"text": "Get a free trial of our premium service!", "label": 1},  # spam
            {"text": "Your package is ready for pickup.", "label": 0}, # not spam
            {"text": "Don't miss out on this exclusive offer!", "label": 1},  # spam
            {"text": "Your account has been verified.", "label": 0}, # not spam
            {"text": "Congratulations! You've been selected for a free gift!", "label": 1},  # spam
            {"text": "Your payment has been processed successfully.", "label": 0},  # not spam
            {"text": "Get a free consultation with our experts!", "label": 1},  # spam
            {"text": "Your account settings have been updated.", "label": 0},  # not spam
            {"text": "Congratulations! You've won a cash prize!", "label": 1},  # spam
            {"text": "Your subscription has been renewed successfully.", "label": 0},  # not spam
            {"text": "Get a free e-book on how to make money online!", "label": 1},  # spam
            {"text": "Your account has been locked due to suspicious activity.", "label": 0},  #not spam
            {"text": "Reminder: Your appointment is tomorrow at 3 PM.", "label": 0},  # not spam
            {'text': 'your invoice is ready for download', 'label': 0},  # not spam
            {"text": "Please find attached the report.", "label": 0},  # not spam
        
        ]
        df = pd.DataFrame(data)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]
        model = LogisticRegression()
        model.fit(X, y)
        return model, vectorizer

    spam_model, spam_vectorizer = train_spam_classifier() 
    label_map = {0: "Not Spam", 1: "Spam"}

    def predict_spam(email_text):
        X_new = spam_vectorizer.transform([email_text])
        pred = spam_model.predict(X_new)[0]
        return pred  # 1 for spam, 0 for not spam


    def classify_inbox_emails():
        SENDER_EMAIL = os.getenv("SENDER_EMAIL")
        EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
        output = []
        with imapclient.IMAPClient('imap.gmail.com') as client:
            client.login(SENDER_EMAIL, EMAIL_PASSWORD)
            client.select_folder("INBOX")
            messages = client.search("ALL")  # Fetch all emails

            for msg_id in messages:
                raw_message = client.fetch([msg_id], ["RFC822"])[msg_id][b"RFC822"]
                email_msg = email.message_from_bytes(raw_message)
                subject = decode_subject(email_msg["Subject"]) if email_msg["Subject"] else "(No Subject)"
                body = extract_email_body(email_msg)
                spam_pred = predict_spam(body)
                output.append(f"Subject: {subject}")
                output.append(f"Prediction: {label_map[spam_pred]}")
                output.append(f"Body (first 100 chars): {body[:100]}\n{'-'*40}")
        return "\n".join(output)

    if __name__ == "__main__":
        classify_inbox_emails()



def main():    
  if __name__ == "__main__":
    print("[DEBUG] Starting Mailing Agent...")
    smtp_server = os.getenv("SMTP_SERVER") or input("Enter SMTP server (e.g., smtp.gmail.com): ")
    port = int(os.getenv("SMTP_PORT") or input("Enter SMTP port (e.g., 465 for SSL): "))
    sender_email = os.getenv("SENDER_EMAIL") or input("Enter sender email: ")
    password = os.getenv("EMAIL_PASSWORD") or input("Enter email password or app-specific password: ")
    receiver_email = os.getenv("RECEIVER_EMAIL") or input("Enter receiver email: ")
    subject = input("Enter the email subject: ")
    attachment_input = input("Enter the full path of the attachment file (or leave blank for none): ")
    attachment_path = attachment_input.strip() if attachment_input.strip() else None

    print("\n[DEBUG] Generating summarized email body via Ollama...... ")
    email_body = generate_body(subject)+"\n\n\n\n\n\n\n[Generated by AI Agent]"
    print("\n[DEBUG] Generated Email Body:\n",'\n', email_body)

   
    # Optional: Provide the option to manually edit the generated summary.
    
    edit_option = input("Would you like to manually edit the generated email body? (yes/no): ").strip().lower()
    if edit_option in ["yes", "y"]:
        print("Enter the new email body below. Press ENTER on an empty line to finish:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        edited_body = "\n".join(lines)
        if edited_body.strip():
            email_body = edited_body+"\n\n\n\n\n\n\n[Generated by AI Agent]"
            print("\n[DEBUG] Updated Email Body:\n", email_body)
        else:
            print("No changes made. Using the generated email body.")

    print("\n[DEBUG] Email details:")
    print("Subject:", subject)
    print()
    print("Email Body:\n", email_body)
    if attachment_path:
        print("Attachment:", attachment_path)
    else:
        print("\nNo attachment provided.\n \n[generated by AI Agent]")

    confirmation = input("\nSend email? (yes/no): ").strip().lower()
    if confirmation in ["yes", "y"]:
        send_email(smtp_server, port, sender_email, password, receiver_email, subject, email_body, attachment_path)
    else:
        print("[DEBUG] Email sending canceled.")



def summary():
# IMAP Credentials
  IMAP_SERVER = "imap.gmail.com"  # Change if using another provider
  EMAIL_ACCOUNT = os.getenv("SENDER_EMAIL")
  PASSWORD = os.getenv("EMAIL_PASSWORD")
  print("[DEBUG] obtaining email content to summarize.....")
# Connect to IMAP & Fetch Emails
  def fetch_emails():
    with imapclient.IMAPClient(IMAP_SERVER) as client:
        client.login(EMAIL_ACCOUNT, PASSWORD)
        client.select_folder("INBOX")
        messages = client.search("UNSEEN")  # Fetch unread emails

        fetched_emails = []
        for msg_id in messages[:5]:  # Fetch last 5 unread emails
            raw_message = client.fetch([msg_id], ["RFC822"])[msg_id][b"RFC822"]
            email_msg = email.message_from_bytes(raw_message)
            
            # Extract Email Body
            body = extract_email_body(email_msg)
            fetched_emails.append(body)
    print("[DEBUG] Fetched emails for summarization.")

    return fetched_emails
    
# Extract text from email (handles plain & HTML)
  def extract_email_body(email_msg):
    if email_msg.is_multipart():
        for part in email_msg.get_payload():
            if part.get_content_type() == "text/plain":  # Extracting plain text
                return part.get_payload(decode=True).decode("utf-8", errors="ignore")
    else:
        return email_msg.get_payload(decode=True).decode("utf-8", errors="ignore")
    
    return "No readable content found."

# Summarization using LangChain & Ollama with PromptTemplate
  def summarize_email(text):
    llm = OllamaLLM(model="qwen2.5-coder:0.5b")  # Change model if needed
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    # Define Prompt Template for structured queries
    prompt_template = PromptTemplate(
        input_variables=["email_content"],
        template="Summarize the following email in a concise manner and in less than 5 lines:\n{email_content}"
    )
    
    formatted_prompt = prompt_template.format(email_content=chunks[0])
    summary = llm.invoke(formatted_prompt)  # Summarizing first chunk
    if not summary.strip():
        summary = "No unseen emails. So no summary generated."
    print("[DEBUG] Generated summary for email.")
    return summary

# Main Execution
  emails = fetch_emails()
  for i, email_text in enumerate(emails):
     print(f"📩 Email {i+1} Summary:\n", summarize_email(email_text))


def foc():
    while True:
        print("\nMailing Agent MARK-II\n")
        print("\nPlease choose an option:\n")
        print("1. Send Email")
        print("2. classify Emails")
        print("3. Summarize Emails")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            main()
        elif choice == "2":
            classifier()
        elif choice == "3":
            summary()
        elif choice == "4":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    foc()