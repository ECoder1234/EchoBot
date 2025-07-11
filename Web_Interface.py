from flask import Flask, render_template, request, jsonify
from EchoBot import get_reply, set_user_persona, session_memory_obj
import os

app = Flask(__name__)

# Initialize the bot
set_user_persona()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({'error': 'Invalid request: JSON body required'}), 400

    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '')
    show_trace = data.get('show_trace', True)

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Get enhanced reply
        reply = get_reply(user_message, show_trace=show_trace)
        
        return jsonify({
            'reply': reply,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': f'Error generating reply: {str(e)}',
            'success': False
        })

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    try:
        session_memory_obj.clear()
        return jsonify({'success': True, 'message': 'Memory cleared'})
    except Exception as e:
        return jsonify({'error': f'Error clearing memory: {str(e)}'})

if __name__ == '__main__':
    print('Running EchoBot web server...')
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🌐 Starting EchoBot Web Interface...")
    print("📱 Open your browser to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 