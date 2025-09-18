import os
import csv
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file
from groq import Groq
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize Groq client with error handling
try:
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        api_key = api_key.strip().strip("'\"")  # Remove quotes if present
        client = Groq(api_key=api_key)
        print(f"Groq client initialized successfully with key: {api_key[:10]}...")
    else:
        print("Warning: GROQ_API_KEY not found in environment variables")
        client = None
except Exception as e:
    print(f"Error initializing Groq client: {str(e)}")
    client = None

def analyze_transcript(transcript):
    try:
        # Check if client is available
        if client is None:
            print("Error: Groq client not initialized")
            return "Error: Groq client not initialized", "Error: Groq client not initialized"

        print("Starting transcript analysis...")

        summary_prompt = f"""
        You are an expert customer service analyst. Summarize the following customer service conversation in exactly 2-3 sentences. Focus on the main issue, actions taken, and outcome.

        Customer Service Conversation:
        {transcript}

        Summary (2-3 sentences only):
        """

        print("Making summary API call...")
        summary_response = client.chat.completions.create(
            messages=[{"role": "user", "content": summary_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )

        summary = summary_response.choices[0].message.content.strip()
        print(f"Summary generated: {summary[:100]}...")

        sentiment_prompt = f"""
        You are an expert sentiment analyst. Analyze the CUSTOMER's overall sentiment throughout this conversation and provide a descriptive sentiment label.

        Customer Service Conversation:
        {transcript}

        Instructions:
        - Focus ONLY on the customer's sentiment, not the agent's
        - Consider the entire conversation, not just the beginning
        - Provide a descriptive sentiment that captures the customer's emotional state

        Choose from these specific sentiment categories:
        - "Satisfied and Positive" - Issue resolved, customer happy/grateful
        - "Frustrated and Negative" - Customer angry, unresolved issues
        - "Confused and Negative" - Customer lost, getting poor help
        - "Disappointed and Negative" - Customer let down by service
        - "Impatient and Negative" - Customer annoyed by delays/process
        - "Relieved and Positive" - Problem solved after difficulty
        - "Grateful and Positive" - Customer appreciative of help
        - "Neutral and Cautious" - Customer uncertain about outcome
        - "Mixed and Neutral" - Customer has both positive and negative feelings

        Customer Sentiment (respond with exactly one phrase from above):
        """

        print("Making sentiment API call...")
        sentiment_response = client.chat.completions.create(
            messages=[{"role": "user", "content": sentiment_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1,
        )

        sentiment = sentiment_response.choices[0].message.content.strip()

        # Clean up sentiment response - remove quotes and extra text
        sentiment_clean = sentiment.replace('"', '').replace("'", "").strip()

        # If the response contains multiple lines, take the first one
        if '\n' in sentiment_clean:
            sentiment_clean = sentiment_clean.split('\n')[0].strip()

        # Extract only the core sentiment phrase (e.g., "Frustrated and Negative")
        # Remove any extra explanatory text after the sentiment
        if ':' in sentiment_clean:
            sentiment_clean = sentiment_clean.split(':')[-1].strip()

        # Take only the first meaningful phrase
        sentiment_parts = sentiment_clean.split('.')
        if len(sentiment_parts) > 1:
            sentiment_clean = sentiment_parts[0].strip()

        print(f"Sentiment analyzed: '{sentiment}' -> cleaned: '{sentiment_clean}'")

        return summary, sentiment_clean

    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        print(error_msg)
        return f"Error generating summary: {str(e)}", f"Error analyzing sentiment: {str(e)}"

def save_to_csv(transcript, summary, sentiment):
    filename = 'call_analysis.csv'

    # Clean the data by replacing newlines with spaces and removing quotes
    clean_transcript = transcript.replace('\n', ' ').replace('\r', ' ').replace('"', '').strip()
    clean_summary = summary.replace('\n', ' ').replace('\r', ' ').replace('"', '').strip()
    clean_sentiment = sentiment.replace('\n', ' ').replace('\r', ' ').replace('"', '').strip()

    # Always create a fresh CSV file (overwrite mode 'w' instead of append 'a')
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Transcript', 'Summary', 'Sentiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)

        # Write header
        writer.writeheader()

        # Write the single row of data
        writer.writerow({
            'Transcript': clean_transcript,
            'Summary': clean_summary,
            'Sentiment': clean_sentiment
        })

    print(f"Fresh CSV created with data: {clean_transcript[:50]}... | {clean_summary[:50]}... | {clean_sentiment}")

@app.route('/')
def index():
    return render_template('index.html')

def parse_json_transcript(json_content):
    try:
        data = json.loads(json_content)

        if isinstance(data, dict):
            if 'transcript' in data:
                return data['transcript']
            elif 'conversation' in data:
                return data['conversation']
            elif 'dialogue' in data:
                return data['dialogue']
            elif 'text' in data:
                return data['text']
            else:
                return str(data)
        elif isinstance(data, list):
            return ' '.join([str(item) for item in data])
        else:
            return str(data)
    except json.JSONDecodeError:
        return None

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        transcript = ''

        # Check if it's a JSON file upload
        if 'json_file' in request.files:
            file = request.files['json_file']
            if file and file.filename.endswith('.json'):
                try:
                    json_content = file.read().decode('utf-8')
                    transcript = parse_json_transcript(json_content)
                    if not transcript:
                        flash('Invalid JSON format. Please check your file.', 'error')
                        return redirect(url_for('index'))
                except Exception as e:
                    flash(f'Error reading JSON file: {str(e)}', 'error')
                    return redirect(url_for('index'))
            else:
                flash('Please upload a valid JSON file.', 'error')
                return redirect(url_for('index'))
        else:
            # Regular text input
            transcript = request.form.get('transcript', '').strip()

        if not transcript:
            flash('Please enter a transcript or upload a JSON file to analyze.', 'error')
            return redirect(url_for('index'))

        summary, sentiment = analyze_transcript(transcript)

        # Only save to CSV if analysis was successful (no errors)
        if not (summary.startswith("Error") or sentiment.startswith("Error")):
            save_to_csv(transcript, summary, sentiment)
        else:
            print("Skipping CSV save due to analysis errors")

        result = {
            'transcript': transcript,
            'summary': summary,
            'sentiment': sentiment
        }

        return render_template('result.html', result=result)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()

    if not data or 'transcript' not in data:
        return jsonify({'error': 'No transcript provided'}), 400

    transcript = data['transcript'].strip()

    if not transcript:
        return jsonify({'error': 'Empty transcript provided'}), 400

    summary, sentiment = analyze_transcript(transcript)

    # Only save to CSV if analysis was successful (no errors)
    if not (summary.startswith("Error") or sentiment.startswith("Error")):
        save_to_csv(transcript, summary, sentiment)
    else:
        print("Skipping CSV save due to analysis errors")

    return jsonify({
        'transcript': transcript,
        'summary': summary,
        'sentiment': sentiment,
        'status': 'success'
    })

@app.route('/history')
def history():
    filename = 'call_analysis.csv'

    if not os.path.isfile(filename):
        return render_template('history.html', records=[])

    records = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        records = list(reader)

    return render_template('history.html', records=records)

@app.route('/download-csv')
def download_csv():
    filename = 'call_analysis.csv'

    if not os.path.isfile(filename):
        # Create empty CSV with just headers if it doesn't exist
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Transcript', 'Summary', 'Sentiment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        flash('No analysis data found. Please analyze a transcript first.', 'error')
        return redirect(url_for('index'))

    try:
        return send_file(
            filename,
            mimetype='text/csv',
            as_attachment=True,
            download_name='call_analysis.csv'
        )
    except Exception as e:
        flash(f'Error downloading CSV: {str(e)}', 'error')
        return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)