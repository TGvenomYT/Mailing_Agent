import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
import threading
from mailing_agent import send_email, generate_body
from main import ollama_query, listen

class CarenChatbotGUI(QWidget):
    append_message_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Caren AI Assistant (Ollama)")
        self.setGeometry(100, 100, 600, 400)
        self.chat_history = []  # Store (sender, message) tuples
        self.init_ui()
        self.append_message_signal.connect(self.append_message)

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e293b, stop:1 #0f172a);
            }
            QTextEdit {
                background-color: #0f172a;
                color: #f1f5f9;
                border-radius: 14px;
                font-size: 16px;
                padding: 12px;
                border: 2px solid #38bdf8;
                font-family: 'Fira Mono', 'Consolas', 'Monaco', monospace;
            }
            QLabel {
                color: #38bdf8;
                font-size: 22px;
                font-weight: bold;
                letter-spacing: 2px;
                padding-bottom: 8px;
            }
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #38bdf8, stop:1 #6366f1);
                color: white;
                border: none;
                border-radius: 14px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #64748b;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(18)

        self.title = QLabel("<h2 style='color:#38bdf8;letter-spacing:2px;margin-bottom:0;'> Caren AI Assistant</h2>")
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Caren is ready to help you!")
        self.chat_display.setMinimumHeight(320)
        self.chat_display.setMaximumHeight(400)
        layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(12)
        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(60)
        self.input_box.setMinimumWidth(300)
        self.input_box.setPlaceholderText("Type your message here...")
        self.input_box.setStyleSheet("font-size:16px;")
        self.input_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout.addWidget(self.input_box)

        self.send_button = QPushButton("Send ✉️")
        self.send_button.setMinimumWidth(90)
        self.send_button.clicked.connect(self.handle_send)
        input_layout.addWidget(self.send_button)

        self.speech_button = QPushButton("🎤 Speak")
        self.speech_button.setMinimumWidth(90)
        self.speech_button.clicked.connect(self.handle_speech)
        input_layout.addWidget(self.speech_button)

        self.mail_button = QPushButton("ComposeEmail")
        self.mail_button.setMinimumWidth(150)
        self.mail_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mail_button.clicked.connect(self.open_mail_dialog)
        input_layout.addWidget(self.mail_button)

        layout.addLayout(input_layout)

        # Add summarize and spam filter buttons in a new row below
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(12)
        self.summary_button = QPushButton("Summarize")
        self.summary_button.setMinimumWidth(90)
        self.summary_button.clicked.connect(self.handle_summarize)
        filter_layout.addWidget(self.summary_button)

        self.spam_button = QPushButton("Spam Filter")
        self.spam_button.setMinimumWidth(90)
        self.spam_button.clicked.connect(self.handle_spam_filter)
        filter_layout.addWidget(self.spam_button)
        layout.addLayout(filter_layout)

        self.setLayout(layout)

    def handle_send(self):
        user_text = self.input_box.toPlainText().strip()
        if not user_text:
            return
        self.append_message_signal.emit("You", user_text)
        self.chat_history.append(("You", user_text))
        self.input_box.clear()
        threading.Thread(target=self.get_ai_response, args=(user_text, False), daemon=True).start()

    def handle_speech(self):
        self.speech_button.setEnabled(False)
        self.append_message_signal.emit("System", "Listening for your voice...")
        threading.Thread(target=self.get_speech_input, daemon=True).start()

    def get_speech_input(self):
        user_text = listen()
        self.speech_button.setEnabled(True)
        if user_text:
            self.append_message_signal.emit("You", user_text)
            self.chat_history.append(("You", user_text))
            threading.Thread(target=self.get_ai_response, args=(user_text, True), daemon=True).start()
        else:
            self.append_message_signal.emit("System", "Sorry, I did not catch that. Please try again.")

    def get_ai_response(self, user_text, speak_response=False):
        # Build a history string for context
        history_str = "\n".join([f"{sender}: {msg}" for sender, msg in self.chat_history[-6:]])  # last 6 exchanges
        from main import ollama_query
        response = ollama_query(user_text, history=history_str)
        self.append_message_signal.emit("Caren", response)
        self.chat_history.append(("Caren", response))
        if speak_response:
            try:
                from main import speak
                speak(response)
            except Exception as e:
                self.append_message_signal.emit("System", f"[Speech error] {e}")

    def append_message(self, sender, message):
        # Remove code block formatting if present
        if message.startswith('```') and message.endswith('```'):
            # Remove the triple backticks and optional language specifier
            lines = message.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            message = '\n'.join(lines)
        self.chat_display.append(f"<b>{sender}:</b> {message}")

    def open_mail_dialog(self):
        from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QTextEdit, QFileDialog, QMessageBox
        from mailing_agent import send_email, generate_body  # Correct import from mailing_agent, not main
        dialog = QDialog(self)
        dialog.setWindowTitle("Compose Email with Caren")
        layout = QFormLayout(dialog)

        to_input = QLineEdit()
        subject_input = QLineEdit()
        body_input = QTextEdit()
        attach_input = QLineEdit()
        attach_btn = QPushButton("Browse...")
        def browse_file():
            file, _ = QFileDialog.getOpenFileName(dialog, "Select Attachment")
            if file:
                attach_input.setText(file)
        attach_btn.clicked.connect(browse_file)

        gen_body_btn = QPushButton("Generate Body with Caren")
        def generate_body_action():
            subject = subject_input.text().strip()
            if not subject:
                QMessageBox.warning(dialog, "Missing Subject", "Please enter a subject to generate the body.")
                return
            gen_body_btn.setEnabled(False)
            try:
                import traceback
                body = generate_body(subject)
                body_input.setPlainText(body)
            except Exception as e:
                tb = traceback.format_exc()
                QMessageBox.critical(dialog, "Error", f"Failed to generate body: {e}\n\nTraceback:\n{tb}")
            finally:
                gen_body_btn.setEnabled(True)

        gen_body_btn.clicked.connect(generate_body_action)

        layout.addRow("To:", to_input)
        layout.addRow("Subject:", subject_input)
        layout.addRow("Body:", body_input)
        layout.addRow(gen_body_btn)
        attach_row = QHBoxLayout()
        attach_row.addWidget(attach_input)
        attach_row.addWidget(attach_btn)
        layout.addRow("Attachment:", attach_row)

        send_btn = QPushButton("Send Email")
        def send_email_action():
            to = to_input.text().strip()
            subject = subject_input.text().strip()
            body = body_input.toPlainText().strip()
            attach = attach_input.text().strip() or None
            if not to or not subject or not body:
                QMessageBox.warning(dialog, "Missing Fields", "Please fill in all required fields.")
                return
            # Use environment or prompt for sender credentials
            import os
            sender = os.getenv("SENDER_EMAIL") or ""
            password = os.getenv("EMAIL_PASSWORD") or ""
            smtp_server = os.getenv("SMTP_SERVER") or "smtp.gmail.com"
            port = int(os.getenv("SMTP_PORT") or 465)
            if not sender or not password:
                QMessageBox.warning(dialog, "Missing Credentials", "Set SENDER_EMAIL and EMAIL_PASSWORD in your .env file.")
                return
            send_email(smtp_server, port, sender, password, to, subject, body, attach)
            QMessageBox.information(dialog, "Success", "Email sent successfully!")
            dialog.accept()
        send_btn.clicked.connect(send_email_action)
        layout.addRow(send_btn)
        dialog.setLayout(layout)
        dialog.exec_()

    def handle_summarize(self):
        # Do not require input text, just run summary() on button click
        self.append_message_signal.emit("System", "Summarizing your Gmail inbox...")
        threading.Thread(target=self.summarize_text, daemon=True).start()

    def summarize_text(self):
        try:
            from mailing_agent import summary
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            summary()  # This will log in to Gmail and summarize unread emails
            sys.stdout = old_stdout
            result = mystdout.getvalue()
            self.append_message_signal.emit("Summary", result)
        except Exception as e:
            self.append_message_signal.emit("System", f"[Summary error] {e}")

    def handle_spam_filter(self):
        self.append_message_signal.emit("System", "Checking your Gmail inbox for spam...")
        threading.Thread(target=self.spam_filter_text, daemon=True).start()

    def spam_filter_text(self):
        try:
            from mailing_agent import classifier
            result = classifier()  # Now returns a string with results
            self.append_message_signal.emit("Spam Filter", result)
        except Exception as e:
            self.append_message_signal.emit("System", f"[Spam filter error] {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarenChatbotGUI()
    window.show()
    sys.exit(app.exec_())
